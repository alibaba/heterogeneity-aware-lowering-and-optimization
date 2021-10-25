//===- odla_tensorrt.cc ---------------------------------------------------===//
//
// Copyright (C) 2019-2021 Alibaba Group Holding Limited.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <NvInferRuntime.h>
#include <NvInferRuntimeCommon.h>
#include <ODLA/odla.h>
#include <bits/stdint-intn.h>
#include <cuda_runtime.h>

#include <cassert>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <unordered_map>
#include <vector>

#include "common.h"
#include "plugins/initPlugin.h"

using namespace nvinfer1;

#if !defined(ODLA_VERSION_NUMBER) || (ODLA_VERSION_NUMBER < 50)
#error This library requires minimum ODLA version 0.5
#endif

struct TrtDestroyer {
  template <typename T>
  void operator()(T* t) {
    if (t) {
      t->destroy();
    }
  }
};

template <typename T>
std::shared_ptr<T> trt_shared_obj(T* obj) {
  return std::shared_ptr<T>(obj, TrtDestroyer());
}

inline bool check(cudaError_t e, int line, const char* file_name) {
  if (e != cudaSuccess) {
    std::cerr << "CUDA runtime API error " << cudaGetErrorName(e) << " at line "
              << line << " in file " << file_name;
    return false;
  }
  return true;
}

inline bool check(bool result, int line, const char* file_name) {
  if (!result) {
    std::cerr << "Error at line " << line << " in file " << file_name;
    return false;
  }
  return true;
}

#define CHECK(call) check(call, __LINE__, __FILE__)

namespace open_dla_tensorrt {
class Logger : public nvinfer1::ILogger {
 public:
  void log(ILogger::Severity severity, const char* msg) NOEXCEPT override {
    int log_level;
    switch (severity) {
      case ILogger::Severity::kINTERNAL_ERROR:
        log_level = 0;
        break;
      case ILogger::Severity::kERROR:
        log_level = 1;
        break;
      case ILogger::Severity::kWARNING:
        log_level = 2;
        break;
      case ILogger::Severity::kINFO:
        log_level = 3;
        break;
      case ILogger::Severity::kVERBOSE:
        log_level = 4;
      default:
        log_level = 5;
    }

    if (log_level <= 1) {
      std::cerr << "[" << log_level << "]: " << msg << "\n";
    }
  }
};
} // namespace open_dla_tensorrt

static open_dla_tensorrt::Logger Logger;

#ifndef NDEBUG
static void LOG_VERBOSE(const std::string& msg) {
  Logger.log(ILogger::Severity::kVERBOSE, msg.c_str());
}

static std::string gen_str(const nvinfer1::Dims& d) {
  std::string s{"("};
  if (d.nbDims != 0) {
    for (int64_t i = 0; i < d.nbDims; i++)
      (s += std::to_string(d.d[i])) += ", ";
    s.pop_back();
    s.pop_back();
  }
  return s + ')';
}
#endif

struct _odla_value {
  _odla_value(nvinfer1::ITensor* tensor, const odla_value_type& type,
              const char* name)
      : tensor(tensor), type(type), name(name) {
    tensor->setName(name);
  }
  _odla_value(nvinfer1::ILayer* layer, const odla_value_type& type,
              const char* name)
      : layer(layer), tensor(layer->getOutput(0)), type(type), name(name) {
    layer->setName(name);
  }

  _odla_value() {}

  operator nvinfer1::ITensor&() { return *tensor; }
  nvinfer1::ILayer* layer = nullptr;
  nvinfer1::ITensor* tensor = nullptr;
  nvinfer1::IConstantLayer* const_layer = nullptr;
  odla_value_type type;
  const char* name;
};

#ifdef MAX_WORKSPACE_SIZE
static constexpr size_t MAX_WORKSPACE_SIZE_BYTES =
    (size_t)MAX_WORKSPACE_SIZE * 1024 * 1024;
#else
static constexpr size_t MAX_WORKSPACE_SIZE_BYTES = 1ul * 1024 * 1024 * 1024;
#endif
static const int MAX_INT64_CONVERTION_NUM = 65536ul;
static bool g_load_engine_mode = false;

struct _odla_computation {
  std::shared_ptr<nvinfer1::IBuilder> builder = nullptr;
  std::shared_ptr<nvinfer1::INetworkDefinition> network = nullptr;
  std::unordered_map<std::string, odla_value> inputs;
  std::unordered_map<std::string, odla_value> outputs;
  std::vector<std::vector<float>> buffers;
  std::vector<std::unique_ptr<_odla_value>> vals;
  std::vector<odla_value> input_vals;
  std::vector<odla_value> output_vals;
  bool fp16_mode = false;

  bool is_dynamic_batch = false;
  int min_batch_size = 0;
  int max_batch_size = 0;
  int opt_batch_size = 0;
  bool load_engine_mode = false;
  size_t max_workspace_size = MAX_WORKSPACE_SIZE_BYTES;

  _odla_computation() {
    load_engine_mode = g_load_engine_mode;
    if (const char* env_p = std::getenv("ODLA_TRT_MAX_WS_MB")) {
      if (int mb = std::stoi(env_p); mb != 0) {
        max_workspace_size = mb << 20;
      }
    }

    if (!load_engine_mode) {
      builder = trt_shared_obj(nvinfer1::createInferBuilder(Logger));
      initODLAPlugin(&Logger, "");
      nvinfer1::NetworkDefinitionCreationFlags flags = 0;
      if (const char* env_p = std::getenv("ODLA_TRT_USE_EXPLICIT_BATCH")) {
        if (*env_p != '0') {
          flags = 1U << static_cast<uint32_t>(
                      nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        }
      }
      network = trt_shared_obj(builder->createNetworkV2(flags));
    }
  }

  ~_odla_computation() {
    builder = nullptr;
    network = nullptr;
  }
};

struct _odla_context {
  odla_computation comp = nullptr;
  std::shared_ptr<nvinfer1::ICudaEngine> engine = nullptr;
  std::shared_ptr<nvinfer1::IExecutionContext> ctx = nullptr;
  std::shared_ptr<nvinfer1::IBuilderConfig> builder_cfg = nullptr;
  nvinfer1::IOptimizationProfile* builder_profile = nullptr;

  typedef struct {
    void* host_ptr = nullptr;
    void* dev_ptr = nullptr;
    size_t len = 0;
    odla_value_type vt;
  } OutputPtrInfo;

  typedef struct {
    const void* host_ptr = nullptr;
    void* dev_ptr = nullptr;
  } InputPtrInfo;
  std::unordered_map<std::string, OutputPtrInfo> output_ptrs;
  std::unordered_map<std::string, InputPtrInfo> input_ptrs;

  int run_batch_size = 0;
  _odla_context(odla_computation comp) : comp(comp) {
    if (!comp->load_engine_mode) {
      builder_cfg = trt_shared_obj(comp->builder->createBuilderConfig());

      if (comp->is_dynamic_batch) {
        builder_profile = comp->builder->createOptimizationProfile();
        for (auto& input : comp->inputs) {
          const char* input_name = input.first.c_str();
          odla_value value = input.second;
          int d1 = value->type.shape.dims[1];
          int d2 = value->type.shape.dims[2];
          int d3 = value->type.shape.dims[3];
          builder_profile->setDimensions(
              input_name, OptProfileSelector::kMIN,
              Dims{4, {comp->min_batch_size, d1, d2, d3}});
          builder_profile->setDimensions(
              input_name, OptProfileSelector::kOPT,
              Dims{4, {comp->opt_batch_size, d1, d2, d3}});
          builder_profile->setDimensions(
              input_name, OptProfileSelector::kMAX,
              Dims{4, {comp->max_batch_size, d1, d2, d3}});
        }
        builder_cfg->addOptimizationProfile(builder_profile);
      }
      builder_cfg->setMaxWorkspaceSize(comp->max_workspace_size);

      if (comp->fp16_mode) {
        builder_cfg->setFlag(BuilderFlag::kFP16);
        builder_cfg->setFlag(BuilderFlag::kSTRICT_TYPES);
      }
      engine = trt_shared_obj(comp->builder->buildEngineWithConfig(
          *comp->network.get(), *builder_cfg.get()));

      ctx = trt_shared_obj(engine->createExecutionContext());
    }
  }
  ~_odla_context() {
    comp = nullptr;
    engine = nullptr;
    ctx = nullptr;
  }
};

struct _odla_executable {
  odla_context context = nullptr;
  odla_computation computation = nullptr;
  int DLACore = -1;
  std::unique_ptr<_odla_value> val;
  _odla_executable(odla_context context, odla_computation computation)
      : context(context), computation(computation) {
    val = std::make_unique<_odla_value>();
  }
  ~_odla_executable() {}
};

static odla_element_type GetODLAType(DataType type) {
  switch (type) {
    case nvinfer1::DataType::kFLOAT:
      return ODLA_FLOAT32;
    case nvinfer1::DataType::kHALF:
      return ODLA_FLOAT16;
    case nvinfer1::DataType::kINT32:
      return ODLA_INT32;
    case nvinfer1::DataType::kINT8:
      return ODLA_INT8;
    case nvinfer1::DataType::kBOOL:
      return ODLA_BOOL;
    default:
      return ODLA_FLOAT32;
  }
}

static int64_t GetTotalElements(const odla_value_shape& dims) {
  return dims.size == 0 ? 1
                        : std::accumulate(dims.dims, dims.dims + dims.size, 1,
                                          std::multiplies<size_t>());
}

const int nvinfer1::Dims::MAX_DIMS;

static nvinfer1::Dims GetNVDims(int n, const odla_uint32* dims) {
  nvinfer1::Dims ret;
  assert(n <= nvinfer1::Dims::MAX_DIMS);
  ret.nbDims = n;
  if (n == 0) {
    ret.d[0] = 0;
  }
  for (int i = 0; i < n; ++i) {
    ret.d[i] = static_cast<int>(dims[i]);
  }
  return ret;
}

static nvinfer1::Dims GetNVDims(const odla_value_shape& dims) {
  nvinfer1::Dims ret;
  ret.nbDims = dims.size;
  assert(dims.size <= std::min(nvinfer1::Dims::MAX_DIMS, ODLA_MAX_DIMENSION));
  if (dims.size == 0) {
    ret.d[0] = 0;
  }
  for (int i = 0; i < dims.size; ++i) {
    ret.d[i] = dims.dims[i];
  }
  return ret;
}

static bool SameNVDims(const nvinfer1::Dims& d1, const nvinfer1::Dims& d2) {
  if (d1.nbDims != d2.nbDims) {
    return false;
  }
  for (int i = 0; i < d1.nbDims; ++i) {
    if (d1.d[i] != d2.d[i]) {
      return false;
    }
  }
  return true;
}

static nvinfer1::Dims BroadcastDims(const odla_value_shape& dims,
                                    size_t dim_size) {
  if (dims.size >= dim_size) {
    return GetNVDims(dims);
  }
  nvinfer1::Dims ret;
  ret.nbDims = dim_size;
  for (int i = 0, e = dim_size - dims.size; i != e; ++i) {
    ret.d[i] = 1;
  }
  for (int i = dim_size - dims.size, j = 0; i != dim_size; ++i, ++j) {
    ret.d[i] = dims.dims[j];
  }
  return ret;
}

static nvinfer1::Dims SqueezeNVDims(const nvinfer1::Dims dims, int index) {
  nvinfer1::Dims ret;
  ret.nbDims = dims.nbDims - 1;
  for (int i = 0, j = 0; i < dims.nbDims; ++i) {
    if (i != index) {
      ret.d[j++] = dims.d[i];
    }
  }
  return ret;
}

thread_local odla_computation g_comp;
static std::vector<std::unique_ptr<_odla_computation>> g_comps;
static std::vector<std::unique_ptr<int[]>> g_workspace;

static nvinfer1::DataType GetNVDataType(odla_element_type type) {
  switch (type) {
    case ODLA_FLOAT32:
      return nvinfer1::DataType::kFLOAT;
    case ODLA_FLOAT16:
      return nvinfer1::DataType::kHALF;
    case ODLA_INT32:
    case ODLA_INT64:
      return nvinfer1::DataType::kINT32;
    case ODLA_INT8:
      return nvinfer1::DataType::kINT8;
    case ODLA_BOOL:
      return nvinfer1::DataType::kBOOL;
    default:
      assert(0 && "unsupported");
      return nvinfer1::DataType::kFLOAT;
  }
}

static unsigned GetElementSize(odla_element_type type) {
  switch (type) {
    case ODLA_FLOAT32:
      return sizeof(float);
    case ODLA_FLOAT16:
      return sizeof(int16_t);
    case ODLA_INT32:
    case ODLA_INT64:
      return sizeof(int32_t);
    case ODLA_INT8:
    case ODLA_BOOL:
      return 1;
    default:
      return 0;
  }
}

static odla_value_type ValidateValueType(const odla_value_type& type) {
  // Trt doesn't support INT64, convert value_type of ODLA_INT64 to ODLA_INT32
  if (type.element_type == ODLA_INT64) {
    return odla_value_type{.element_type = ODLA_INT32, .shape = type.shape};
  }
  return type;
}

static std::unique_ptr<int[]> ConvertData(const odla_value_type& type,
                                          const void* ptr) {
  if (type.element_type == ODLA_INT64) {
    const int64_t* src = static_cast<const int64_t*>(ptr);
    auto num_elements = GetTotalElements(type.shape);
    auto buf = std::make_unique<int[]>(num_elements);
    int* tmp = buf.get();
    for (int i = 0; i < num_elements; ++i) {
      assert(*src < MAX_INT64_CONVERTION_NUM);
      tmp[i] = (static_cast<int>(*src++));
    }
    return buf;
  }
  return nullptr;
}

template <typename T>
static odla_value CreateValue(T* t, const odla_value_type& type,
                              const odla_value_id id) {
  const char* name = reinterpret_cast<const char*>(id);

  auto v = std::make_unique<_odla_value>(t, type, name);
  auto ret = v.get();
  g_comp->vals.push_back(std::move(v));
  return ret;
}

static std::vector<nvinfer1::RNNGateType> GetRNNGateOrder(
    const odla_rnn_gate_order gate_order) {
  const nvinfer1::RNNGateType trt_gate_iofc[] = {
      nvinfer1::RNNGateType::kINPUT, nvinfer1::RNNGateType::kOUTPUT,
      nvinfer1::RNNGateType::kFORGET, nvinfer1::RNNGateType::kCELL};
  const nvinfer1::RNNGateType trt_gate_ifco[] = {
      nvinfer1::RNNGateType::kINPUT, nvinfer1::RNNGateType::kFORGET,
      nvinfer1::RNNGateType::kCELL, nvinfer1::RNNGateType::kOUTPUT};
  const nvinfer1::RNNGateType trt_gate_ifoc[] = {
      nvinfer1::RNNGateType::kINPUT, nvinfer1::RNNGateType::kFORGET,
      nvinfer1::RNNGateType::kOUTPUT, nvinfer1::RNNGateType::kCELL};
  const nvinfer1::RNNGateType trt_gate_icof[] = {
      nvinfer1::RNNGateType::kINPUT, nvinfer1::RNNGateType::kCELL,
      nvinfer1::RNNGateType::kOUTPUT, nvinfer1::RNNGateType::kFORGET};

  switch (gate_order) {
    case ODLA_RNN_IFCO:
      return std::vector<nvinfer1::RNNGateType>(trt_gate_ifco,
                                                trt_gate_ifco + 4);
    case ODLA_RNN_IFOC:
      return std::vector<nvinfer1::RNNGateType>(trt_gate_ifoc,
                                                trt_gate_ifoc + 4);
    case ODLA_RNN_ICOF:
      return std::vector<nvinfer1::RNNGateType>(trt_gate_icof,
                                                trt_gate_icof + 4);
    default:
      // default order as iofc
      return std::vector<nvinfer1::RNNGateType>(trt_gate_iofc,
                                                trt_gate_iofc + 4);
  }
}

extern "C" {
odla_status odla_CreateComputation(odla_computation* computation) {
  g_comps.push_back(std::make_unique<_odla_computation>());
  g_comp = g_comps.back().get();
  *computation = g_comp;
  return ODLA_SUCCESS;
}

odla_status odla_SetActiveComputation(odla_computation computation) {
  g_comp = computation;
  return ODLA_SUCCESS;
}

odla_status odla_DestroyComputation(odla_computation comp) {
  for (auto it = g_comps.begin(), e = g_comps.end(); it != e; ++it) {
    if (it->get() == comp) {
      it->reset();
      g_comps.erase(it);
      return ODLA_SUCCESS;
    }
  }
  assert(0);
  return ODLA_FAILURE;
}

odla_status odla_SetComputationItem(odla_computation computation,
                                    odla_item_type type,
                                    odla_item_value value) {
  bool is_dynamic_batch = false;
  switch (type) {
    case ODLA_DYNAMIC_BATCH:
      is_dynamic_batch = *(reinterpret_cast<bool*>(value));
      if (is_dynamic_batch &&
          (computation->is_dynamic_batch != is_dynamic_batch)) {
        computation->is_dynamic_batch = is_dynamic_batch;
        if (!computation->load_engine_mode) {
          computation->network.reset();
          nvinfer1::NetworkDefinitionCreationFlags flags =
              1U << static_cast<uint32_t>(
                  nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
          computation->network =
              trt_shared_obj(computation->builder->createNetworkV2(flags));
        }
      }
      break;

    case ODLA_MIN_BATCH_SIZE:
      computation->min_batch_size = *(reinterpret_cast<int*>(value));
      break;

    case ODLA_MAX_BATCH_SIZE:
      computation->max_batch_size = *(reinterpret_cast<int*>(value));
      break;

    case ODLA_OPT_BATCH_SIZE:
      computation->opt_batch_size = *(reinterpret_cast<int*>(value));
      break;

    case ODLA_FP16_MODE:
      computation->fp16_mode = *(reinterpret_cast<bool*>(value));
      break;

    case ODLA_LOAD_ENGINE_MODE:
      g_load_engine_mode = *(reinterpret_cast<bool*>(value));
      break;

    case ODLA_BF16_MODE:
      break;

    default:
      std::cerr << "Unsupported property type: " << type << std::endl;
      return ODLA_FAILURE;
  }

  return ODLA_SUCCESS;
}

odla_status odla_SetContextItem(odla_context context, odla_item_type type,
                                odla_item_value value) {
  switch (type) {
    case ODLA_RUN_BATCH_SIZE:
      context->run_batch_size = *(reinterpret_cast<int*>(value));
      break;

    default:
      std::cerr << "Unsupported property type: " << type << std::endl;
      return ODLA_FAILURE;
  }

  return ODLA_SUCCESS;
}

odla_status odla_CreateContext(odla_context* context) {
  *context = new _odla_context(g_comp);
  return ODLA_SUCCESS;
}

odla_status odla_DestroyContext(odla_context context) {
  delete context;
  return ODLA_SUCCESS;
}

odla_value odla_CreateArgument(odla_value_type type, const odla_value_id id) {
  const char* name = reinterpret_cast<const char*>(id);
  auto input = g_comp->network->addInput(name, GetNVDataType(type.element_type),
                                         GetNVDims(type.shape));
  odla_value v = CreateValue(input, type, id);
  g_comp->inputs[name] = v;
  g_comp->input_vals.push_back(v);
  return v;
}

odla_status odla_GetNumOfArgsFromComputation(const odla_computation computation,
                                             odla_uint32* num_args) {
  *num_args = computation->input_vals.size();
  return ODLA_SUCCESS;
}

odla_status odla_GetArgFromComputationByIdx(const odla_computation computation,
                                            const odla_uint32 arg_idx,
                                            odla_value* arg_value) {
  *arg_value = nullptr;
  if (arg_idx >= computation->input_vals.size()) {
    return ODLA_FAILURE;
  }
  *arg_value = computation->input_vals[arg_idx];
  return ODLA_SUCCESS;
}

odla_value odla_CreateConstant(odla_value_type type, const void* ptr,
                               const odla_value_id id) {
  void* host_ptr = const_cast<void*>(ptr);
  auto buf = ConvertData(type, ptr);
  if (buf != nullptr) {
    host_ptr = buf.get();
    g_workspace.push_back(std::move(buf));
  }
  nvinfer1::Weights weight{.type = GetNVDataType(type.element_type),
                           .values = host_ptr,
                           .count = GetTotalElements(type.shape)};
  auto c = g_comp->network->addConstant(GetNVDims(type.shape), weight);
  odla_value v = CreateValue(c->getOutput(0), ValidateValueType(type), id);
  v->const_layer = c;
  return v;
}

odla_status odla_SetValueAsOutput(const odla_value val) {
  const char* name =
      val->layer != nullptr ? val->layer->getName() : val->tensor->getName();
  g_comp->outputs[name] = val;
  g_comp->output_vals.push_back(val);
  val->tensor->setName(name);
  val->tensor->setType(GetNVDataType(val->type.element_type));
  g_comp->network->markOutput(*val->tensor);
  return ODLA_SUCCESS;
}
odla_status odla_GetNumOfOutputsFromComputation(
    const odla_computation computation, odla_uint32* num_outputs) {
  *num_outputs = computation->output_vals.size();
  return ODLA_SUCCESS;
}

odla_status odla_GetOutputFromComputationByIdx(
    const odla_computation computation, const odla_uint32 output_idx,
    odla_value* output_value) {
  *output_value = nullptr;
  if (output_idx >= computation->output_vals.size()) {
    return ODLA_FAILURE;
  }
  *output_value = computation->output_vals[output_idx];
  return ODLA_SUCCESS;
}

odla_status odla_BindToArgument(odla_value value, const odla_void* data_ptr,
                                odla_context context) {
  odla_value_shape real_shape = value->type.shape;
  bool dynamic_input_size = false;
  if ((g_comp && g_comp->is_dynamic_batch) || context->run_batch_size) {
    real_shape.dims[0] = context->run_batch_size;
    dynamic_input_size = true;
  }
  size_t bytes =
      GetTotalElements(real_shape) * GetElementSize(value->type.element_type);
  void* validated_data_ptr = const_cast<void*>(data_ptr);
  auto buf = ConvertData(value->type, data_ptr);
  if (buf != nullptr) {
    validated_data_ptr = buf.get();
  }
  void* dev_ptr = context->input_ptrs[value->name].dev_ptr;
  if (dev_ptr == nullptr) {
    CHECK(cudaMalloc(&dev_ptr, bytes));
  }
  context->input_ptrs[value->name] = {.host_ptr = data_ptr, .dev_ptr = dev_ptr};

  CHECK(cudaMemcpy(dev_ptr, validated_data_ptr, bytes, cudaMemcpyHostToDevice));

  return ODLA_SUCCESS;
}

odla_status odla_BindToArgumentById(const odla_value_id value_id,
                                    const odla_void* data_ptr,
                                    odla_context context) {
  std::string name((const char*)value_id);

  return odla_BindToArgument(context->comp->inputs[name], data_ptr, context);
}

odla_status odla_BindToOutput(odla_value value, odla_void* data_ptr,
                              odla_context context) {
  odla_value_shape real_shape = value->type.shape;
  if ((g_comp && g_comp->is_dynamic_batch) || context->run_batch_size) {
    real_shape.dims[0] = context->run_batch_size;
  }
  size_t bytes =
      GetTotalElements(real_shape) * GetElementSize(value->type.element_type);
  // TODO: convert to int64 for int64 outputs?
  void* dst = context->output_ptrs[value->name].dev_ptr;
  if (dst == nullptr) {
    CHECK(cudaMalloc(&dst, bytes));
  }

  context->output_ptrs[value->name] = {
      .host_ptr = data_ptr, .dev_ptr = dst, .len = bytes, .vt = value->type};

  return ODLA_SUCCESS;
}

odla_status odla_BindToOutputById(const odla_value_id value_id,
                                  odla_void* data_ptr, odla_context context) {
  std::string name((const char*)value_id);

  assert(context->comp->outputs.count(name));
  auto val = context->comp->outputs[name];
  return odla_BindToOutput(val, data_ptr, context);
}

static odla_status odla_StoreEngine(odla_context context,
                                    const odla_char* file_name) {
  if (context == nullptr) {
    return ODLA_FAILURE;
  }

  std::string engine = file_name;
  std::ofstream engineFile(engine, std::ios::binary);
  if (!engineFile) {
    std::cerr << "Cannot open engine file: " << engine << std::endl;
    return ODLA_FAILURE;
  }

  std::shared_ptr<IHostMemory> serializedEngine{context->engine->serialize()};
  if (serializedEngine == nullptr) {
    std::cerr << "Engine serialization failed" << std::endl;
    return ODLA_FAILURE;
  }

  engineFile.write(static_cast<char*>(serializedEngine->data()),
                   serializedEngine->size());

  if (engineFile.fail()) {
    return ODLA_FAILURE;
  }

  return ODLA_SUCCESS;
}

odla_status odla_StoreExecutable(const odla_char* file_name,
                                 const odla_executable executable) {
  odla_context Ctx = executable->context;
  return odla_StoreEngine(Ctx, file_name);
}

static odla_status odla_LoadEngine(odla_context context,
                                   const odla_char* file_name, int DLACore) {
  std::ifstream engineFile(file_name, std::ios::binary);
  if (!engineFile) {
    std::cerr << "Error opening engine file: " << file_name << std::endl;
    return ODLA_FAILURE;
  }

  engineFile.seekg(0, engineFile.end);
  long int fsize = engineFile.tellg();
  engineFile.seekg(0, engineFile.beg);

  std::vector<char> engineData(fsize);
  engineFile.read(engineData.data(), fsize);
  if (!engineFile) {
    std::cerr << "Error loading engine file: " << file_name << std::endl;
    return ODLA_FAILURE;
  }

  std::shared_ptr<IRuntime> runtime{createInferRuntime(Logger)};
  if (DLACore != -1) {
    runtime->setDLACore(DLACore);
  }

  context->engine = trt_shared_obj(
      runtime->deserializeCudaEngine(engineData.data(), fsize, nullptr));
  context->ctx = trt_shared_obj(context->engine->createExecutionContext());

  return ODLA_SUCCESS;
}

odla_status odla_LoadExecutable(const odla_char* file_name,
                                odla_executable* executable,
                                odla_context* context,
                                odla_computation* computation) {
  int DLACore = (*executable)->DLACore;

  if (*computation == nullptr) {
    int load_engine_mode = 1;
    odla_SetComputationItem(nullptr, ODLA_LOAD_ENGINE_MODE,
                            (odla_item_value)&load_engine_mode);
    odla_CreateComputation(computation);
    bool is_dynamic_batch = true;
    odla_SetComputationItem(*computation, ODLA_DYNAMIC_BATCH,
                            (odla_item_value)&is_dynamic_batch);
  }
  if (*context == nullptr) {
    odla_CreateContext(context);
  };

  if (odla_LoadEngine(*context, file_name, DLACore) != ODLA_SUCCESS) {
    return ODLA_FAILURE;
  }

  (*executable)->computation = *computation;
  (*executable)->context = *context;

  return ODLA_SUCCESS;
}

odla_status odla_GetNumOfOutputsFromExecutable(const odla_executable executable,
                                               odla_uint32* num_outputs) {
  odla_context Ctx = executable->context;
  int numBindings = Ctx->engine->getNbBindings();
  int numOutput = 0;
  for (int i = 0; i < numBindings; ++i) {
    if (!Ctx->engine->bindingIsInput(i)) {
      numOutput++;
    }
  }
  *num_outputs = numOutput;
  return ODLA_SUCCESS;
}

odla_status odla_GetNumOfArgsFromExecutable(const odla_executable executable,
                                            odla_uint32* num_args) {
  odla_context Ctx = executable->context;
  int numBindings = Ctx->engine->getNbBindings();
  int numInput = 0;
  for (int i = 0; i < numBindings; ++i) {
    if (Ctx->engine->bindingIsInput(i)) {
      numInput++;
    }
  }
  *num_args = numInput;
  return ODLA_SUCCESS;
}

odla_status odla_CreateExecutable(odla_executable* executable,
                                  odla_context context,
                                  odla_computation computation) {
  *executable = new _odla_executable(context, computation);
  return ODLA_SUCCESS;
}

odla_status odla_DestroyExecutable(odla_executable executable) {
  delete executable;
  return ODLA_SUCCESS;
}

static odla_status odla_GetValFromExecutableByIdx(
    const odla_executable executable, const odla_uint32 idx,
    odla_value* value) {
  odla_context Ctx = executable->context;
  auto val = executable->val.get();
  Dims dims = Ctx->engine->getBindingDimensions(idx);
  DataType nv_type = Ctx->engine->getBindingDataType(idx);
  auto type = &val->type;
  type->element_type = GetODLAType(nv_type);
  type->shape.size = dims.nbDims;
  for (int i = 0; i < dims.nbDims; ++i) {
    type->shape.dims[i] = dims.d[i];
  }
  // val->tensor->setName(Ctx->engine->getBindingName(idx));
  val->name = Ctx->engine->getBindingName(idx);
  *value = val;
  return ODLA_SUCCESS;
}

odla_status odla_GetArgFromExecutableByIdx(const odla_executable executable,
                                           const odla_uint32 arg_idx,
                                           odla_value* arg_value) {
  odla_context Ctx = executable->context;
  auto getIdxInEngine = [&Ctx, arg_idx] {
    int numBindings = Ctx->engine->getNbBindings();
    for (int i = 0, j = 0; i < numBindings; ++i) {
      if (Ctx->engine->bindingIsInput(i)) {
        if (j == arg_idx) {
          return i;
        }
        j++;
      }
    }
  };

  return odla_GetValFromExecutableByIdx(executable, getIdxInEngine(),
                                        arg_value);
}

odla_status ODLA_API_CALL odla_GetOutputFromExecutableByIdx(
    const odla_executable executable, const odla_uint32 output_idx,
    odla_value* output_value) {
  odla_context Ctx = executable->context;
  auto getIdxInEngine = [&Ctx, output_idx] {
    int numBindings = Ctx->engine->getNbBindings();
    for (int i = 0, j = 0; i < numBindings; ++i) {
      if (!Ctx->engine->bindingIsInput(i)) {
        if (j == output_idx) {
          return i;
        }
        j++;
      }
    }
  };

  return odla_GetValFromExecutableByIdx(executable, getIdxInEngine(),
                                        output_value);
}

odla_status odla_GetValueType(const odla_value value,
                              odla_value_type* value_type) {
  *value_type = value->type;
  return ODLA_SUCCESS;
}

odla_status odla_ExecuteComputation(odla_computation comp, odla_context context,
                                    odla_compute_mode mode,
                                    odla_device device) {
  std::vector<void*> buffers;
  auto add_to_buffer = [&](const std::string& name, void* ptr) {
    int idx = context->engine->getBindingIndex(name.c_str());
    if (idx >= 0) {
      if (buffers.size() <= idx) {
        buffers.resize(idx + 1);
      }
      buffers[idx] = ptr;
    }
  };
  for (auto& kv : context->input_ptrs) {
    add_to_buffer(kv.first, kv.second.dev_ptr);
  }
  for (auto& kv : context->output_ptrs) {
    add_to_buffer(kv.first, kv.second.dev_ptr);
  }
  if (comp->is_dynamic_batch) {
    for (auto& input_ptr : context->input_ptrs) {
      int idx = context->engine->getBindingIndex(input_ptr.first.c_str());
      nvinfer1::Dims dims = context->ctx->getBindingDimensions(idx);
      dims.d[0] = context->run_batch_size;
      context->ctx->setBindingDimensions(idx, dims);
    }
    CHECK(context->ctx->executeV2(buffers.data()));
  } else {
    int batch = 1;
    CHECK(context->ctx->execute(batch, buffers.data()));
  }
  for (auto& kv : context->output_ptrs) {
    if (kv.second.vt.element_type == ODLA_INT64) {
      std::vector<int> host_tmp(GetTotalElements(kv.second.vt.shape));
      CHECK(cudaMemcpy(host_tmp.data(), kv.second.dev_ptr, kv.second.len,
                       cudaMemcpyDeviceToHost));
      int64_t* ptr = static_cast<int64_t*>(kv.second.host_ptr);
      for (int d : host_tmp) {
        *ptr++ = static_cast<int64_t>(d);
      }
    } else {
      CHECK(cudaMemcpy(kv.second.host_ptr, kv.second.dev_ptr, kv.second.len,
                       cudaMemcpyDeviceToHost));
    }
  }
  if (!comp->is_dynamic_batch) {
    return ODLA_SUCCESS;
  }
  // copy results and free temp buffers.
  for (auto& ptr : buffers) {
    CHECK(cudaFree(ptr));
  }

  context->input_ptrs.clear();
  context->output_ptrs.clear();
  return ODLA_SUCCESS;
}

static odla_value_shape broadcastTensor(odla_computation comp,
                                        nvinfer1::ITensor*& lhs,
                                        nvinfer1::ITensor*& rhs,
                                        odla_value_shape dims_lhs,
                                        odla_value_shape dims_rhs) {
  if (dims_lhs.size == dims_rhs.size) {
    return dims_lhs;
  }
  if (dims_lhs.size > dims_rhs.size) {
    auto reshape = g_comp->network->addShuffle(*rhs);
    reshape->setReshapeDimensions(BroadcastDims(dims_rhs, dims_lhs.size));
    rhs = reshape->getOutput(0);
    return dims_lhs;
  }
  auto reshape = g_comp->network->addShuffle(*lhs);
  reshape->setReshapeDimensions(BroadcastDims(dims_lhs, dims_rhs.size));
  lhs = reshape->getOutput(0);
  return dims_rhs;
}

static odla_value binary_op(nvinfer1::ElementWiseOperation op, odla_value lhs,
                            odla_value rhs, const odla_value_id id) {
  nvinfer1::ITensor* lhs_tensor = lhs->tensor;
  nvinfer1::ITensor* rhs_tensor = rhs->tensor;
  const auto& dims_lhs = lhs->type.shape;
  const auto& dims_rhs = rhs->type.shape;
  auto out_dim =
      broadcastTensor(g_comp, lhs_tensor, rhs_tensor, dims_lhs, dims_rhs);
  auto sub = g_comp->network->addElementWise(*lhs_tensor, *rhs_tensor, op);
  for (int i = 0; i < out_dim.size; ++i) {
    out_dim.dims[i] = std::max(lhs_tensor->getDimensions().d[i],
                               rhs_tensor->getDimensions().d[i]);
  }
  auto ret_type = lhs->type.element_type;
  if (op == nvinfer1::ElementWiseOperation::kEQUAL ||
      op == nvinfer1::ElementWiseOperation::kGREATER ||
      op == nvinfer1::ElementWiseOperation::kLESS) {
    ret_type = ODLA_BOOL;
  }
  return CreateValue(sub, {ret_type, out_dim}, id);
}

odla_value odla_Add(odla_value lhs, odla_value rhs, const odla_value_id id) {
  return binary_op(nvinfer1::ElementWiseOperation::kSUM, lhs, rhs, id);
}

odla_value odla_And(odla_value lhs, odla_value rhs, const odla_value_id id) {
  return binary_op(nvinfer1::ElementWiseOperation::kAND, lhs, rhs, id);
}

odla_value odla_Sub(odla_value lhs, odla_value rhs, const odla_value_id id) {
  return binary_op(nvinfer1::ElementWiseOperation::kSUB, lhs, rhs, id);
}

odla_value odla_Mul(odla_value lhs, odla_value rhs, const odla_value_id id) {
  return binary_op(nvinfer1::ElementWiseOperation::kPROD, lhs, rhs, id);
}

odla_value odla_Div(odla_value lhs, odla_value rhs, const odla_value_id id) {
  return binary_op(nvinfer1::ElementWiseOperation::kDIV, lhs, rhs, id);
}

odla_value odla_Equal(odla_value lhs, odla_value rhs, const odla_value_id id) {
  return binary_op(nvinfer1::ElementWiseOperation::kEQUAL, lhs, rhs, id);
}

odla_value odla_Or(odla_value lhs, odla_value rhs, const odla_value_id id) {
  return binary_op(nvinfer1::ElementWiseOperation::kOR, lhs, rhs, id);
}

odla_value odla_NotEqual(odla_value lhs, odla_value rhs,
                         const odla_value_id id) {
  auto eq = odla_Equal(lhs, rhs, nullptr);
  return odla_Not(eq, id);
}

odla_value odla_Greater(odla_value lhs, odla_value rhs,
                        const odla_value_id id) {
  return binary_op(nvinfer1::ElementWiseOperation::kGREATER, lhs, rhs, id);
}

odla_value odla_GreaterOrEqual(odla_value lhs, odla_value rhs,
                               const odla_value_id id) {
  auto gt = odla_Greater(lhs, rhs, nullptr);
  auto eq = odla_Equal(lhs, rhs, nullptr);
  return odla_Or(gt, eq, id);
}

odla_value odla_Less(odla_value lhs, odla_value rhs, const odla_value_id id) {
  return binary_op(nvinfer1::ElementWiseOperation::kLESS, lhs, rhs, id);
}

odla_value odla_LessOrEqual(odla_value lhs, odla_value rhs,
                            const odla_value_id id) {
  auto gt = odla_Less(lhs, rhs, nullptr);
  auto eq = odla_Equal(lhs, rhs, nullptr);
  return odla_Or(gt, eq, id);
}

static odla_value unary_op(nvinfer1::UnaryOperation op, odla_value input,
                           const odla_value_id id) {
  auto layer = g_comp->network->addUnary(*input->tensor, op);
  return CreateValue(layer, input->type, id);
}

odla_value odla_Log(odla_value input, const odla_value_id id) {
  return unary_op(nvinfer1::UnaryOperation::kLOG, input, id);
}

odla_value odla_Not(odla_value input, const odla_value_id id) {
  return unary_op(nvinfer1::UnaryOperation::kNOT, input, id);
}

odla_value odla_Abs(odla_value input, const odla_value_id id) {
  return unary_op(nvinfer1::UnaryOperation::kABS, input, id);
}

odla_value odla_Max(odla_value lhs, odla_value rhs, const odla_value_id id) {
  return binary_op(nvinfer1::ElementWiseOperation::kMAX, lhs, rhs, id);
}

odla_value odla_Min(odla_value lhs, odla_value rhs, const odla_value_id id) {
  return binary_op(nvinfer1::ElementWiseOperation::kMIN, lhs, rhs, id);
}

odla_value odla_Cast(odla_value input, odla_element_type target_type,
                     const odla_value_id id) {
  auto op = g_comp->network->addIdentity(*input);
  odla_value_type dst_type{target_type, input->type.shape};
  op->setOutputType(0, GetNVDataType(dst_type.element_type));
  return CreateValue(op, dst_type, id);
}

odla_value odla_Ceil(odla_value input, const odla_value_id id) {
  return unary_op(nvinfer1::UnaryOperation::kCEIL, input, id);
}

odla_value odla_Floor(odla_value input, const odla_value_id id) {
  return unary_op(nvinfer1::UnaryOperation::kFLOOR, input, id);
}

odla_value odla_Erf(odla_value input, const odla_value_id id) {
  return unary_op(nvinfer1::UnaryOperation::kERF, input, id);
}

odla_value odla_Exp(odla_value input, const odla_value_id id) {
  return unary_op(nvinfer1::UnaryOperation::kEXP, input, id);
}

odla_value odla_Sqrt(odla_value input, const odla_value_id id) {
  return unary_op(nvinfer1::UnaryOperation::kSQRT, input, id);
}

odla_value odla_Rsqrt(odla_value input, const odla_value_id id) {
  auto op = g_comp->network->addUnary(*input, nvinfer1::UnaryOperation::kSQRT);
  op = g_comp->network->addUnary(*(op->getOutput(0)),
                                 nvinfer1::UnaryOperation::kRECIP);
  return CreateValue(op, input->type, id);
}

odla_value odla_Sin(odla_value input, const odla_value_id id) {
  return unary_op(nvinfer1::UnaryOperation::kSIN, input, id);
}

odla_value odla_Sinh(odla_value input, const odla_value_id id) {
  return unary_op(nvinfer1::UnaryOperation::kSINH, input, id);
}

odla_value odla_Cos(odla_value input, const odla_value_id id) {
  return unary_op(nvinfer1::UnaryOperation::kCOS, input, id);
}

odla_value odla_Cosh(odla_value input, const odla_value_id id) {
  return unary_op(nvinfer1::UnaryOperation::kCOSH, input, id);
}

odla_value odla_Tan(odla_value input, const odla_value_id id) {
  auto op0 = g_comp->network->addUnary(*input, nvinfer1::UnaryOperation::kSIN);
  auto op1 = g_comp->network->addUnary(*input, nvinfer1::UnaryOperation::kCOS);
  auto op = g_comp->network->addElementWise(
      *(op0->getOutput(0)), *(op1->getOutput(0)),
      nvinfer1::ElementWiseOperation::kDIV);
  return CreateValue(op, input->type, id);
}

odla_value odla_Tanh(odla_value input, const odla_value_id id) {
  auto op =
      g_comp->network->addActivation(*input, nvinfer1::ActivationType::kTANH);
  return CreateValue(op, input->type, id);
}

odla_value odla_ACos(odla_value input, const odla_value_id id) {
  return unary_op(nvinfer1::UnaryOperation::kACOS, input, id);
}

odla_value odla_ACosh(odla_value input, const odla_value_id id) {
  return unary_op(nvinfer1::UnaryOperation::kACOSH, input, id);
}

odla_value odla_ASin(odla_value input, const odla_value_id id) {
  return unary_op(nvinfer1::UnaryOperation::kASIN, input, id);
}

odla_value odla_ASinh(odla_value input, const odla_value_id id) {
  return unary_op(nvinfer1::UnaryOperation::kASINH, input, id);
}

odla_value odla_ATan(odla_value input, const odla_value_id id) {
  return unary_op(nvinfer1::UnaryOperation::kATAN, input, id);
}

odla_value odla_ATanh(odla_value input, const odla_value_id id) {
  return unary_op(nvinfer1::UnaryOperation::kATANH, input, id);
}

odla_value odla_Neg(odla_value input, const odla_value_id id) {
  return unary_op(nvinfer1::UnaryOperation::kNEG, input, id);
}

odla_value odla_Pad(odla_value input, const odla_uint32* padding_front,
                    const odla_uint32* padding_back,
                    odla_value_shape output_dims, const odla_value_id id) {
  const auto& input_dims = input->type.shape;
  assert(input_dims.size >= 3 && input_dims.dims[0] == output_dims.dims[0]);
#if NV_TENSORRT_MAJOR < 7
  auto pad = g_comp->network->addPadding(
      *input->tensor,
      nvinfer1::DimsHW{static_cast<int>(padding_front[2]),
                       static_cast<int>(padding_front[3])},
      nvinfer1::DimsHW{static_cast<int>(padding_back[2]),
                       static_cast<int>(padding_back[3])});
#else
  auto pad = g_comp->network->addPaddingNd(*input->tensor,
                                           GetNVDims(2, padding_front + 2),
                                           GetNVDims(2, padding_back + 2));
#endif
  return CreateValue(pad, {input->type.element_type, output_dims}, id);
}

odla_value odla_Clamp(odla_value input, odla_float32 lo, odla_float32 hi,
                      const odla_value_id id) {
  auto relu =
      g_comp->network->addActivation(*input, nvinfer1::ActivationType::kCLIP);
  relu->setAlpha(lo);
  relu->setBeta(hi);
  return CreateValue(relu, input->type, id);
}

odla_value odla_Relu(odla_value input, const odla_value_id id) {
  auto relu =
      g_comp->network->addActivation(*input, nvinfer1::ActivationType::kRELU);
  return CreateValue(relu, input->type, id);
}

odla_value odla_LeakyRelu(odla_value input, odla_float32 alpha,
                          const odla_value_id id) {
  auto relu = g_comp->network->addActivation(
      *input, nvinfer1::ActivationType::kLEAKY_RELU);
  relu->setAlpha(alpha);
  return CreateValue(relu, input->type, id);
}

odla_value odla_ThresholdedRelu(odla_value input, odla_float32 alpha,
                                const odla_value_id id) {
  auto relu = g_comp->network->addActivation(
      *input, nvinfer1::ActivationType::kTHRESHOLDED_RELU);
  relu->setAlpha(alpha);
  return CreateValue(relu, input->type, id);
}

odla_value odla_Selu(odla_value input, odla_float32 alpha, odla_float32 beta,
                     const odla_value_id id) {
  auto relu =
      g_comp->network->addActivation(*input, nvinfer1::ActivationType::kSELU);
  relu->setAlpha(alpha);
  relu->setBeta(beta);
  return CreateValue(relu, input->type, id);
}

odla_value odla_Elu(odla_value input, odla_float32 alpha,
                    const odla_value_id id) {
  auto elu =
      g_comp->network->addActivation(*input, nvinfer1::ActivationType::kELU);
  elu->setAlpha(alpha);
  return CreateValue(elu, input->type, id);
}

odla_value odla_PRelu(odla_value input, odla_value slope,
                      const odla_value_id value_id) {
  // prelu(input, slope) = (input < 0 ? input * slope : input)
  //                     = relu(input) + min(max(input,0), -INF) * slope
  std::string name_stem = std::string(reinterpret_cast<const char*>(value_id));
  auto name = name_stem + "_relu";
  auto relu = odla_Relu(input, (odla_value_id)name.data());
  name = name_stem + "_neg";
  auto neg_input = odla_Clamp(input, std::numeric_limits<float>::lowest(), 0,
                              (odla_value_id)name.data());
  name = name_stem + "_rhs";
  auto neg_prelu = odla_Mul(neg_input, slope, (odla_value_id)name.data());
  return odla_Add(relu, neg_prelu, value_id);
}

odla_value odla_Resize(odla_value input, odla_interpolation_mode interpolation,
                       odla_resize_coordinate_mode mode, odla_uint32 axes_mask,
                       odla_value_shape output_dims,
                       const odla_value_id value_id) {
  assert(interpolation == ODLA_NEAREST);
  auto resize = g_comp->network->addResize(*input);
  resize->setResizeMode(nvinfer1::ResizeMode::kNEAREST);
  resize->setOutputDimensions(GetNVDims(output_dims));
  return CreateValue(resize, {input->type.element_type, output_dims}, value_id);
}

odla_value odla_Softmax(odla_value input, odla_int32 axis,
                        const odla_value_id id) {
  const auto& dims = input->type.shape;
  auto sm = g_comp->network->addSoftMax(*input);
  axis = axis < 0 ? dims.size - 1 : axis;
  sm->setAxes(1 << axis);
  return CreateValue(sm, input->type, id);
}

odla_value odla_Sigmoid(odla_value input, const odla_value_id value_id) {
  auto relu = g_comp->network->addActivation(
      *input, nvinfer1::ActivationType::kSIGMOID);
  return CreateValue(relu, input->type, value_id);
}

static odla_value reduce(odla_value input, nvinfer1::ReduceOperation op,
                         odla_size_t num_of_axes, const odla_uint32* axes,
                         odla_bool keep_dims, odla_value_shape output_dims,
                         const odla_value_id id) {
  if (output_dims.size != input->type.shape.size) {
    assert(!keep_dims);
  }

  uint32_t reduce_axes = 0;
  for (int i = 0; i < num_of_axes; ++i) {
    reduce_axes |= (1 << axes[i]);
  }

  auto ret = g_comp->network->addReduce(*input, op, reduce_axes, keep_dims);
  return CreateValue(ret, {input->type.element_type, output_dims}, id);
}

odla_value odla_ReduceL1(odla_value input, odla_size_t num_of_axes,
                         const odla_uint32* axes, odla_bool keep_dims,
                         odla_value_shape output_dims, const odla_value_id id) {
  const auto& name = std::string(reinterpret_cast<const char*>(id)) + "_extra";
  return reduce(odla_Abs(input, (const odla_value_id)name.c_str()),
                nvinfer1::ReduceOperation::kSUM, num_of_axes, axes, keep_dims,
                output_dims, id);
}

odla_value odla_ReduceL2(odla_value input, odla_size_t num_of_axes,
                         const odla_uint32* axes, odla_bool keep_dims,
                         odla_value_shape output_dims, const odla_value_id id) {
  const auto& name = std::string(reinterpret_cast<const char*>(id)) + "_extra";
  return odla_Sqrt(
      odla_ReduceSumSquare(input, num_of_axes, axes, keep_dims, output_dims,
                           (const odla_value_id)name.c_str()),
      id);
}

odla_value odla_ReduceLogSum(odla_value input, odla_size_t num_of_axes,
                             const odla_uint32* axes, odla_bool keep_dims,
                             odla_value_shape output_dims,
                             const odla_value_id id) {
  const auto& name = std::string(reinterpret_cast<const char*>(id)) + "_extra";
  return odla_Log(
      odla_ReduceSum(input, num_of_axes, axes, keep_dims, output_dims,
                     (const odla_value_id)name.c_str()),
      id);
}

odla_value odla_ReduceLogSumExp(odla_value input, odla_size_t num_of_axes,
                                const odla_uint32* axes, odla_bool keep_dims,
                                odla_value_shape output_dims,
                                const odla_value_id id) {
  const auto& name = std::string(reinterpret_cast<const char*>(id));
  auto reduce_max =
      odla_ReduceMax(input, num_of_axes, axes, true, output_dims,
                     (const odla_value_id)(name + "_extra_1").c_str());
  auto exp_delta =
      odla_Exp(odla_Sub(input, reduce_max,
                        (const odla_value_id)(name + "_extra2").c_str()),
               (const odla_value_id)(name + "_extra3").c_str());
  auto reduce_max_keep_dims =
      odla_ReduceMax(input, num_of_axes, axes, keep_dims, output_dims,
                     (const odla_value_id)(name + "_extra4").c_str());
  return odla_Add(
      odla_ReduceLogSum(exp_delta, num_of_axes, axes, keep_dims, output_dims,
                        (const odla_value_id)(name + "_extra5").c_str()),
      reduce_max_keep_dims, id);
}

odla_value odla_ReduceMean(odla_value input, odla_size_t num_of_axes,
                           const odla_uint32* axes, odla_bool keep_dims,
                           odla_value_shape output_dims,
                           const odla_value_id id) {
  return reduce(input, nvinfer1::ReduceOperation::kAVG, num_of_axes, axes,
                keep_dims, output_dims, id);
}

odla_value odla_ReduceMin(odla_value input, odla_size_t num_of_axes,
                          const odla_uint32* axes, odla_bool keep_dims,
                          odla_value_shape output_dims,
                          const odla_value_id id) {
  return reduce(input, nvinfer1::ReduceOperation::kMIN, num_of_axes, axes,
                keep_dims, output_dims, id);
}

odla_value odla_ReduceMax(odla_value input, odla_size_t num_of_axes,
                          const odla_uint32* axes, odla_bool keep_dims,
                          odla_value_shape output_dims,
                          const odla_value_id id) {
  return reduce(input, nvinfer1::ReduceOperation::kMAX, num_of_axes, axes,
                keep_dims, output_dims, id);
}

odla_value odla_ReduceSum(odla_value input, odla_size_t num_of_axes,
                          const odla_uint32* axes, odla_bool keep_dims,
                          odla_value_shape output_dims,
                          const odla_value_id id) {
  return reduce(input, nvinfer1::ReduceOperation::kSUM, num_of_axes, axes,
                keep_dims, output_dims, id);
}

odla_value odla_ReduceSumSquare(odla_value input, odla_size_t num_of_axes,
                                const odla_uint32* axes, odla_bool keep_dims,
                                odla_value_shape output_dims,
                                const odla_value_id id) {
  const auto& name = std::string(reinterpret_cast<const char*>(id)) + "_extra";
  return reduce(odla_Mul(input, input, (const odla_value_id)name.c_str()),
                nvinfer1::ReduceOperation::kSUM, num_of_axes, axes, keep_dims,
                output_dims, id);
}

odla_value odla_ReduceProd(odla_value input, odla_size_t num_of_axes,
                           const odla_uint32* axes, odla_bool keep_dims,
                           odla_value_shape output_dims,
                           const odla_value_id id) {
  return reduce(input, nvinfer1::ReduceOperation::kPROD, num_of_axes, axes,
                keep_dims, output_dims, id);
}

odla_value odla_LRN(odla_value input, odla_memory_layout input_layout,
                    odla_int32 window_size, odla_float32 alpha,
                    odla_float32 beta, odla_float32 bias,
                    const odla_value_id value_id) {
  auto const& type = input->type.element_type;
  const auto& input_dims = input->type.shape;
  assert(input_layout == odla_memory_layout::ODLA_CHANNELS_FIRST);
  assert(type == ODLA_FLOAT32);
  auto lrn = g_comp->network->addLRN(*input, window_size, alpha, beta, bias);
  return CreateValue(lrn, input->type, value_id);
}

odla_value odla_BatchNormalization(odla_value input,
                                   odla_memory_layout input_layout,
                                   odla_value mean, odla_value var,
                                   odla_float32 epsilon, odla_value scale,
                                   odla_value offset, odla_float32 scalar_scale,
                                   odla_float32 scalar_offset,
                                   const odla_value_id value_id) {
  auto const& type = input->type.element_type;
  const auto& input_dims = input->type.shape;
  assert(type == ODLA_FLOAT32);

  int channel_index = (input_layout == odla_memory_layout::ODLA_CHANNELS_FIRST)
                          ? 1
                          : input_dims.size - 1;
  int64_t C = input_dims.dims[channel_index];

  g_comp->buffers.push_back(std::vector<float>(C));
  g_comp->buffers.push_back(std::vector<float>(C));
  auto& shift_buf = g_comp->buffers[g_comp->buffers.size() - 2];
  auto& multiply_buf = g_comp->buffers.back();

  assert(mean->const_layer);
  assert(var->const_layer);

  const float* mean_data =
      static_cast<const float*>(mean->const_layer->getWeights().values);
  const float* var_data =
      static_cast<const float*>(var->const_layer->getWeights().values);
  const float* scale_data = static_cast<const float*>(
      scale && scale->const_layer ? scale->const_layer->getWeights().values
                                  : nullptr);
  const float* offset_data = static_cast<const float*>(
      offset && offset->const_layer ? offset->const_layer->getWeights().values
                                    : nullptr);

  for (int64_t i = 0; i < C; ++i) {
    multiply_buf[i] = 1.0f / std::sqrt(var_data[i] + epsilon);
    float s = (scale_data) ? scale_data[i] : scalar_scale;
    multiply_buf[i] *= s;
    float b = offset_data ? offset_data[i] : scalar_offset;
    shift_buf[i] = -mean_data[i] * multiply_buf[i] + b;
  }

  nvinfer1::Weights shift{
      .type = GetNVDataType(type), .values = shift_buf.data(), .count = C};
  nvinfer1::Weights multiply{
      .type = GetNVDataType(type), .values = multiply_buf.data(), .count = C};
  nvinfer1::Weights power{
      .type = GetNVDataType(type), .values = nullptr, .count = 0};

  auto bn = g_comp->network->addScale(*input, nvinfer1::ScaleMode::kCHANNEL,
                                      shift, multiply, power);
  bn->setChannelAxis(channel_index);
  return CreateValue(bn, input->type, value_id);
}

odla_value odla_InstanceNormalization(
    odla_value input, odla_memory_layout input_layout, odla_float32 epsilon,
    odla_value scale, odla_value offset, odla_float32 scalar_scale,
    odla_float32 scalar_offset, const odla_value_id value_id) {
  std::vector<nvinfer1::ITensor*> inputs = {input->tensor, scale->tensor,
                                            offset->tensor};
  const static char* plugin_name = "InstanceNormalization_TRT";
  const static char* plugin_ver = "001";
  auto creator = getPluginRegistry()->getPluginCreator(plugin_name, plugin_ver);
  std::vector<nvinfer1::PluginField> f;
  int nb_chs = input->type.shape.dims[1];
  f.emplace_back("epsilon", &epsilon, nvinfer1::PluginFieldType::kFLOAT32, 1);
  assert(scale->const_layer != nullptr && offset->const_layer != nullptr);
  f.emplace_back("scales", scale->const_layer->getWeights().values,
                 nvinfer1::PluginFieldType::kFLOAT32, nb_chs);
  f.emplace_back("bias", offset->const_layer->getWeights().values,
                 nvinfer1::PluginFieldType::kFLOAT32, nb_chs);

  nvinfer1::PluginFieldCollection plugin_data;
  plugin_data.nbFields = f.size();
  plugin_data.fields = f.data();
  auto plugin = creator->createPlugin(plugin_name, &plugin_data);
  auto norm = g_comp->network->addPluginV2(
      &inputs[0], static_cast<int>(inputs.size()), *plugin);
  return CreateValue(norm->getOutput(0), input->type, value_id);
}

odla_value odla_Conv(odla_value input, odla_memory_layout input_layout,
                     odla_uint32 group, odla_value kernel,
                     odla_memory_layout kernel_layout,
                     const odla_uint32* strides, const odla_uint32* dilations,
                     const odla_uint32* paddings_front,
                     const odla_uint32* paddings_back, odla_value bias,
                     odla_value_shape output_dims, const odla_value_id id) {
  auto kernel_weights = kernel->const_layer->getWeights();
  const auto& kernel_dims = kernel->type.shape;
  nvinfer1::Weights bias_weights{kernel_weights.type, nullptr, 0};
  int oc = output_dims.dims[1];
#if NV_TENSORRT_MAJOR < 7
  auto conv = g_comp->network->addConvolution(
      *input, oc,
      nvinfer1::DimsHW{static_cast<int>(kernel_dims.dims[2]),
                       static_cast<int>(kernel_dims.dims[3])},
      kernel_weights, bias_weights);
  conv->setStride(nvinfer1::DimsHW(strides[0], strides[1]));
#else
  auto conv = g_comp->network->addConvolutionNd(
      *input, oc,
      nvinfer1::DimsHW{static_cast<int>(kernel_dims.dims[2]),
                       static_cast<int>(kernel_dims.dims[3])},
      kernel_weights, bias_weights);
  conv->setStrideNd(nvinfer1::DimsHW(strides[0], strides[1]));
#endif
  conv->setPrePadding(nvinfer1::DimsHW(paddings_front[0], paddings_front[1]));
  conv->setPostPadding(nvinfer1::DimsHW(paddings_back[0], paddings_back[1]));
  odla_value_type output_type{.element_type = input->type.element_type,
                              .shape = output_dims};
  if (group > 1) {
    conv->setNbGroups(static_cast<int>(group));
  }
  auto ret = CreateValue(conv, output_type, id);
  if (bias && bias->const_layer &&
      GetTotalElements(bias->type.shape) == output_dims.dims[1]) {
    conv->setBiasWeights(bias->const_layer->getWeights());
    return ret;
  }
  return bias ? odla_Add(ret, bias, id) : ret;
}

odla_value odla_DeConv(odla_value input, odla_memory_layout input_layout,
                       odla_uint32 group, odla_value kernel,
                       odla_memory_layout kernel_layout,
                       const odla_uint32* strides, const odla_uint32* dilations,
                       const odla_uint32* paddings_front,
                       const odla_uint32* paddings_back, odla_value bias,
                       odla_value_shape output_dims, const odla_value_id id) {
  auto kernel_dims = kernel->type.shape;
  auto kernel_weights = kernel->const_layer->getWeights();
  nvinfer1::Weights bias_weights{kernel_weights.type, nullptr, 0};
  if (bias != nullptr) {
    bias_weights = bias->const_layer->getWeights();
  }

  int oc = output_dims.dims[1];

#if NV_TENSORRT_MAJOR < 7
  auto conv = g_comp->network->addDeconvolution(
      *input, oc,
      nvinfer1::DimsHW{static_cast<int>(kernel_dims.dims[2]),
                       static_cast<int>(kernel_dims.dims[3])},
      kernel_weights, bias_weights);
  conv->setStride(nvinfer1::DimsHW(strides[0], strides[1]));
#else
  auto conv = g_comp->network->addDeconvolutionNd(
      *input, oc,
      nvinfer1::DimsHW{static_cast<int>(kernel_dims.dims[2]),
                       static_cast<int>(kernel_dims.dims[3])},
      kernel_weights, bias_weights);
  conv->setStrideNd(nvinfer1::DimsHW(strides[0], strides[1]));
#endif

  conv->setPrePadding(nvinfer1::DimsHW(paddings_front[0], paddings_front[1]));
  conv->setPostPadding(nvinfer1::DimsHW(paddings_back[0], paddings_back[1]));
  conv->setNbGroups(static_cast<int>(group));
  odla_value_type output_type{.element_type = input->type.element_type,
                              .shape = output_dims};
  return CreateValue(conv, output_type, id);
}

odla_value odla_Concat(odla_values inputs, odla_int32 axis,
                       odla_value_shape output_dims, const odla_value_id id) {
  int num = inputs.size;
  std::vector<nvinfer1::ITensor*> input_tensors(num);
  for (int i = 0; i < num; ++i) {
    input_tensors[i] = inputs.values[i]->tensor;
  }

  auto concat = g_comp->network->addConcatenation(input_tensors.data(), num);
  concat->setAxis(axis);
  odla_value_type output_type{
      .element_type = inputs.values[0]->type.element_type,
      .shape = output_dims};
  return CreateValue(concat, output_type, id);
}

odla_value odla_MaxPool(odla_value input, odla_memory_layout input_layout,
                        const odla_uint32* window_dims,
                        const odla_uint32* strides,
                        const odla_uint32* paddings_front,
                        const odla_uint32* paddings_back,
                        odla_value_shape output_dims,
                        const odla_value_id value_id) {
  assert(input_layout == odla_memory_layout::ODLA_CHANNELS_FIRST);
#if NV_TENSORRT_MAJOR < 7
  auto pooling = g_comp->network->addPooling(
      *input, nvinfer1::PoolingType::kMAX,
      nvinfer1::DimsHW(window_dims[0], window_dims[1]));
  pooling->setStride(nvinfer1::DimsHW(strides[0], strides[1]));
  pooling->setPrePadding(
      nvinfer1::DimsHW(paddings_front[0], paddings_front[1]));
  pooling->setPostPadding(nvinfer1::DimsHW(paddings_back[0], paddings_back[1]));
#else
  auto pooling = g_comp->network->addPoolingNd(
      *input, nvinfer1::PoolingType::kMAX,
      nvinfer1::DimsHW(window_dims[0], window_dims[1]));
  pooling->setStrideNd(nvinfer1::DimsHW(strides[0], strides[1]));
  pooling->setPrePadding(
      nvinfer1::DimsHW(paddings_front[0], paddings_front[1]));
  pooling->setPostPadding(nvinfer1::DimsHW(paddings_back[0], paddings_back[1]));
#endif

  odla_value_type output_type{.element_type = input->type.element_type,
                              .shape = output_dims};

  return CreateValue(pooling, output_type, value_id);
}

odla_value odla_AveragePool(odla_value input, odla_memory_layout input_layout,
                            const odla_uint32* window_dims,
                            const odla_uint32* strides,
                            const odla_uint32* paddings_front,
                            const odla_uint32* paddings_back,
                            odla_value_shape output_dims,
                            const odla_value_id value_id) {
  assert(input_layout == odla_memory_layout::ODLA_CHANNELS_FIRST);
#if NV_TENSORRT_MAJOR < 7
  auto pooling = g_comp->network->addPooling(
      *input, nvinfer1::PoolingType::kAVERAGE,
      nvinfer1::DimsHW(window_dims[0], window_dims[1]));
  pooling->setStride(nvinfer1::DimsHW(strides[0], strides[1]));
  pooling->setPrePadding(
      nvinfer1::DimsHW(paddings_front[0], paddings_front[1]));
  pooling->setPostPadding(nvinfer1::DimsHW(paddings_back[0], paddings_back[1]));
#else
  auto pooling = g_comp->network->addPoolingNd(
      *input, nvinfer1::PoolingType::kAVERAGE,
      nvinfer1::DimsHW(window_dims[0], window_dims[1]));
  pooling->setStrideNd(nvinfer1::DimsHW(strides[0], strides[1]));
  pooling->setPrePadding(
      nvinfer1::DimsHW(paddings_front[0], paddings_front[1]));
  pooling->setPostPadding(nvinfer1::DimsHW(paddings_back[0], paddings_back[1]));
#endif

  odla_value_type output_type{.element_type = input->type.element_type,
                              .shape = output_dims};

  return CreateValue(pooling, output_type, value_id);
}

odla_value odla_Fill(odla_value_type type, odla_fill_method method,
                     odla_float32 p0, odla_float32 p1, odla_float32 seed,
                     const odla_value_id value_id) {
  if (method == ODLA_RandomUniform) {
    auto fill = g_comp->network->addFill(
        GetNVDims(type.shape), nvinfer1::FillOperation::kRANDOM_UNIFORM);
    fill->setAlpha(p0);
    fill->setBeta(p1);
    return CreateValue(fill, type, value_id);
  }
  assert(0 && "Unsupported fill method");
}

odla_value odla_Gemm(odla_value lhs, odla_bool transpose_lhs, odla_value rhs,
                     odla_bool transpose_rhs, odla_float32 alpha,
                     odla_float32 beta, odla_value bias,
                     odla_value_shape output_dims, const odla_value_id id) {
  nvinfer1::ILayer* fc = nullptr;
  const auto& lhs_dims = lhs->type.shape;
  if (!transpose_lhs && transpose_rhs && rhs->const_layer) {
    auto input = lhs->tensor;
    if (lhs_dims.size == 2) {
      auto reshape = g_comp->network->addShuffle(*lhs);
      odla_value_shape dim{.size = 4,
                           {lhs_dims.dims[0], lhs_dims.dims[1], 1, 1}};
      reshape->setReshapeDimensions(GetNVDims(dim));
      input = reshape->getOutput(0);
    }
    auto const& kernel_weights = rhs->const_layer->getWeights();
    nvinfer1::Weights bias_weights{kernel_weights.type, nullptr, 0};
    if (bias) bias_weights = bias->const_layer->getWeights();
    fc = g_comp->network->addFullyConnected(*input, output_dims.dims[1],
                                            rhs->const_layer->getWeights(),
                                            bias_weights);
    auto reshape = g_comp->network->addShuffle(*fc->getOutput(0));
    reshape->setReshapeDimensions(GetNVDims(output_dims));
    fc = reshape;
  } else {
    auto getOp = [](bool trans) {
      return trans ? nvinfer1::MatrixOperation::kTRANSPOSE
                   : nvinfer1::MatrixOperation::kNONE;
    };
    fc = g_comp->network->addMatrixMultiply(*lhs, getOp(transpose_lhs), *rhs,
                                            getOp(transpose_rhs));
    if (bias) {
      fc = g_comp->network->addElementWise(
          *fc->getOutput(0), *bias, nvinfer1::ElementWiseOperation::kSUM);
    }
  }
  return CreateValue(fc, odla_value_type{lhs->type.element_type, output_dims},
                     id);
}

odla_value odla_Reshape(odla_value input, odla_value_shape output_dims,
                        const odla_value_id id) {
  auto shuffle = g_comp->network->addShuffle(*input);
  shuffle->setReshapeDimensions(GetNVDims(output_dims));
  return CreateValue(shuffle, {input->type.element_type, output_dims}, id);
}

odla_value odla_Transpose(odla_value input, odla_value_shape permutations,
                          odla_value_shape output_dims,
                          const odla_value_id id) {
  auto shuffle = g_comp->network->addShuffle(*input);

  nvinfer1::Permutation perm;
  for (int i = 0, e = permutations.size; i < e; ++i)
    perm.order[i] = permutations.dims[i];
  shuffle->setFirstTranspose(perm);
  shuffle->setReshapeDimensions(GetNVDims(output_dims));
  return CreateValue(shuffle, {input->type.element_type, output_dims}, id);
}

odla_value odla_ArgMax(odla_value input, odla_int32 axis, odla_bool keep_dims,
                       odla_bool return_last_index,
                       odla_value_type output_value_type,
                       const odla_value_id id) {
  unsigned reduce_axes = axis < 0 ? input->type.shape.size + axis : axis;
  auto topk = g_comp->network->addTopK(*input, nvinfer1::TopKOperation::kMAX, 1,
                                       1 << reduce_axes);
  return CreateValue(topk->getOutput(1), output_value_type, id);
}

odla_value odla_ArgMin(odla_value input, odla_int32 axis, odla_bool keep_dims,
                       odla_bool return_last_index,
                       odla_value_type output_value_type,
                       const odla_value_id id) {
  unsigned reduce_axes = axis < 0 ? input->type.shape.size + axis : axis;
  auto topk = g_comp->network->addTopK(*input, nvinfer1::TopKOperation::kMIN, 1,
                                       1 << reduce_axes);
  return CreateValue(topk->getOutput(1), output_value_type, id);
}

odla_value odla_Gather(odla_value input, const odla_value indices,
                       odla_int32 axis, odla_value_shape output_dims,
                       const odla_value_id id) {
  axis = axis < 0 ? input->type.shape.size - 1 : axis;
  assert(indices->type.element_type == ODLA_INT32 ||
         indices->type.element_type == ODLA_INT64);
  auto input_t = input->tensor;
  if (input->type.element_type == ODLA_BOOL) {
    const auto& name = std::string(input->name) + "_cast";
    input_t =
        odla_Cast(input, ODLA_INT32, (const odla_value_id)name.c_str())->tensor;
  }
  auto gather = g_comp->network->addGather(*input_t, *indices, axis);
  if (input->type.element_type == ODLA_BOOL) {
    const auto& gather_name =
        std::string(reinterpret_cast<const char*>(id)) + "_extra";
    auto gather_v =
        CreateValue(gather, odla_value_type{ODLA_INT32, output_dims},
                    (const odla_value_id)gather_name.c_str());
    g_comp->buffers.push_back(std::vector<float>(1, 0.0));
    const auto& zero_name = std::string(gather_name) + "_comp_zero";
    auto zero_v = odla_CreateConstant(
        odla_value_type{ODLA_INT32, odla_value_shape{0, {}}},
        g_comp->buffers.back().data(), (const odla_value_id)zero_name.c_str());
    return odla_Greater(gather_v, zero_v, id);
  }
  return CreateValue(gather, {input->type.element_type, output_dims}, id);
}

odla_value odla_Slice(odla_value input, const odla_uint32* start,
                      const odla_uint32* end, const odla_uint32* stride,
                      odla_value_shape output_dims, const odla_value_id id) {
  odla_value_shape start_dims, stride_dims;
  const auto& dims = input->type.shape;
  start_dims.size = dims.size;
  stride_dims.size = dims.size;
  for (int i = 0; i < dims.size; ++i) {
    start_dims.dims[i] = start[i];
    stride_dims.dims[i] = stride[i];
  }
  auto slice =
      g_comp->network->addSlice(*input, GetNVDims(start_dims),
                                GetNVDims(output_dims), GetNVDims(stride_dims));
  return CreateValue(slice, {input->type.element_type, output_dims}, id);
}

odla_value odla_NMS(odla_value boxes, odla_value scores,
                    odla_uint32 max_num_outputs, odla_float32 iou_threshold,
                    odla_float32 score_threshold,
                    odla_value_type output_value_type,
                    const odla_value_id value_id) {
  const static int num_inputs = 2;
  bool shared_location = true;
  auto input_boxes = boxes->tensor;
  if (shared_location) {
    const auto& dims = boxes->type.shape;
    assert(dims.size == 3);
    odla_value_shape new_shape{
        .size = 3, .dims = {dims.dims[0] * dims.dims[1], 1, dims.dims[2]}};
    auto boxes_reshape = g_comp->network->addShuffle(*boxes->tensor);
    boxes_reshape->setReshapeDimensions(GetNVDims(new_shape));
    input_boxes = boxes_reshape->getOutput(0);
  }

  // nmsPlugin requires scores of shape [batch_size,
  // num_boxes, num_classes]
  const auto& dims = scores->type.shape;
  assert(dims.size == 3);
  int num_classes = dims.dims[1];
  odla_value_shape new_shape{
      .size = 2, .dims = {dims.dims[0] * dims.dims[2], dims.dims[1]}};
  auto score_trans = g_comp->network->addShuffle(*scores);
  nvinfer1::Permutation perm;
  perm.order[0] = 0;
  perm.order[1] = 2;
  perm.order[2] = 1;
  score_trans->setFirstTranspose(perm);
  score_trans->setReshapeDimensions(GetNVDims(new_shape));

  std::vector<nvinfer1::ITensor*> inputs = {input_boxes,
                                            score_trans->getOutput(0)};
  nvinfer1::plugin::NMSParameters param{
      .shareLocation = shared_location,
      .backgroundLabelId = -1,
      .numClasses = num_classes,
      .topK = static_cast<int>(max_num_outputs),
      .keepTopK = static_cast<int>(max_num_outputs),
      .scoreThreshold = score_threshold,
      .iouThreshold = iou_threshold,
      .isNormalized = true,
  };
#if NV_TENSORRT_MAJOR < 7
  auto nms_plugin = createBatchedNMSPlugin(param);
#else
  auto creator =
      getPluginRegistry()->getPluginCreator("BatchedNMS_TRT_V2", "1");
  nvinfer1::PluginFieldCollection pluginData;
  std::vector<nvinfer1::PluginField> f;
  f.emplace_back("shareLocation", &param.shareLocation,
                 nvinfer1::PluginFieldType::kINT32, 1);
  f.emplace_back("backgroundLabelId", &param.backgroundLabelId,
                 nvinfer1::PluginFieldType::kINT32, 1);
  f.emplace_back("numClasses", &param.numClasses,
                 nvinfer1::PluginFieldType::kINT32, 1);
  f.emplace_back("topK", &param.topK, nvinfer1::PluginFieldType::kINT32, 1);
  f.emplace_back("keepTopK", &param.keepTopK, nvinfer1::PluginFieldType::kINT32,
                 1);
  f.emplace_back("scoreThreshold", &param.scoreThreshold,
                 nvinfer1::PluginFieldType::kFLOAT32, 1);
  f.emplace_back("iouThreshold", &param.iouThreshold,
                 nvinfer1::PluginFieldType::kFLOAT32, 1);
  f.emplace_back("isNormalized", &param.isNormalized,
                 nvinfer1::PluginFieldType::kINT32, 1);
  pluginData.nbFields = f.size();
  pluginData.fields = f.data();
  auto nms_plugin = creator->createPlugin("BatchedNMS_TRT", &pluginData);
#endif

  auto nms = g_comp->network->addPluginV2(&inputs[0], num_inputs, *nms_plugin);
  return CreateValue(nms->getOutput(4), output_value_type, value_id);
}

odla_value odla_Tile(odla_value input, const odla_uint32* repeat,
                     odla_value_shape output_dims,
                     const odla_value_id value_id) {
  auto dims = input->type.shape.size;
  nvinfer1::Dims start{.nbDims = dims};
  nvinfer1::Dims stride{.nbDims = dims};
  nvinfer1::Dims size{.nbDims = dims};
  for (int i = 0; i != dims; ++i) {
    start.d[i] = 0;
    stride.d[i] = 1;
    size.d[i] = repeat[i] * input->type.shape.dims[i];
  }
  auto op = g_comp->network->addSlice(*input, start, size, stride);
  op->setMode(nvinfer1::SliceMode::kWRAP);
  return CreateValue(op, {input->type.element_type, output_dims}, value_id);
}

odla_values odla_TopK(odla_value input, odla_uint32 K, odla_bool largest,
                      odla_bool sorted, odla_uint32 axis,
                      odla_value_type output_value_type,
                      odla_value_type output_value_index_type,
                      const odla_value_ids value_ids) {
  nvinfer1::TopKOperation op = (largest == true)
                                   ? nvinfer1::TopKOperation::kMAX
                                   : nvinfer1::TopKOperation::kMIN;
  auto topk = g_comp->network->addTopK(*input, op, K, 1 << axis);
  return {.size = 2,
          .values = {CreateValue(topk->getOutput(0), output_value_type,
                                 value_ids.value_ids[0]),
                     CreateValue(topk->getOutput(1), output_value_index_type,
                                 value_ids.value_ids[1])}};
}

odla_value odla_Pow(odla_value base, odla_value exponent,
                    const odla_value_id value_id) {
  return binary_op(nvinfer1::ElementWiseOperation::kPOW, base, exponent,
                   value_id);
}

odla_value odla_Reciprocal(odla_value input, const odla_value_id value_id) {
  return unary_op(nvinfer1::UnaryOperation::kRECIP, input, value_id);
}

#ifndef REPLACE_RNN_WITH_LOOP
odla_values odla_LSTM(odla_value input, odla_rnn_weight_format weight_format,
                      odla_rnn_gate_order gate_order,
                      odla_value_shape weight_dims, odla_value W, odla_value R,
                      odla_value B, odla_value sequence_lens,
                      odla_value initial_h, odla_value initial_c, odla_value P,
                      odla_int32 hidden_size, odla_rnn_direction direction,
                      odla_rnn_outputs outputs, const odla_value_ids value_id) {
  const int num_layers = 1;
  const int num_gates = 4;
  const int seq_size = input->type.shape.dims[0];
  const int batch_size = input->type.shape.dims[1];
  const int input_size = input->type.shape.dims[2];
  const int num_directions = (direction == ODLA_RNN_BIDIRECTIONAL) ? 2 : 1;
  const auto rnn_dir = (direction == ODLA_RNN_BIDIRECTIONAL)
                           ? nvinfer1::RNNDirection::kBIDIRECTION
                           : nvinfer1::RNNDirection::kUNIDIRECTION;

  auto input_t = input->tensor;
  // input layout [seq, batch, input]
  // trt assume [batch, seq, input]
  auto transpose_layer = g_comp->network->addShuffle(*input_t);
  nvinfer1::Permutation perm{1, 0, 2};
  transpose_layer->setFirstTranspose(perm);
  transpose_layer->setReshapeDimensions(
      nvinfer1::Dims{3, {batch_size, seq_size, input_size}});
  input_t = transpose_layer->getOutput(0);
  nvinfer1::IRNNv2Layer* rnn_layer =
      g_comp->network->addRNNv2(*input_t, num_layers, hidden_size, seq_size,
                                nvinfer1::RNNOperation::kLSTM);
  rnn_layer->setDirection(rnn_dir);

  // prepare initial hidden and initial cell
  auto getInitTensor = [&num_directions, &hidden_size,
                        &batch_size](odla_value init_v) -> nvinfer1::ITensor* {
    if (init_v) {
      return init_v->tensor;
    }
    odla_value_shape dim{.size = 3,
                         .dims = {batch_size, num_directions, hidden_size}};
    g_comp->buffers.push_back(std::vector<float>(GetTotalElements(dim), 0.0f));
    nvinfer1::Weights weight{.type = nvinfer1::DataType::kFLOAT,
                             .values = g_comp->buffers.back().data(),
                             .count = GetTotalElements(dim)};
    return g_comp->network->addConstant(GetNVDims(dim), weight)->getOutput(0);
  };
  nvinfer1::ITensor* init_hidden_t = getInitTensor(initial_h);
  // LOG_VERBOSE("init_hidden dim:" +
  // gen_str(init_hidden_t->getDimensions()));
  nvinfer1::ITensor* init_cell_t = getInitTensor(initial_c);
  rnn_layer->setHiddenState(*init_hidden_t);
  rnn_layer->setCellState(*init_cell_t);

  // weight order [iofc]
  assert(W->const_layer != nullptr && R->const_layer != nullptr);
  nvinfer1::Weights input_w = W->const_layer->getWeights();
  nvinfer1::Weights recurrence_w = R->const_layer->getWeights();
  const auto& trt_gate_orders = GetRNNGateOrder(gate_order);
  size_t offset_w = 0, offset_bias = 0;
  for (int gate_index = 0; gate_index < 2 * num_gates; ++gate_index) {
    bool isW = (gate_index < num_gates);
    int64_t weight_count = (isW ? input_size : hidden_size) * hidden_size;
    const float* weight_ptr =
        isW ? static_cast<const float*>(input_w.values)
            : static_cast<const float*>(recurrence_w.values);
    nvinfer1::Weights gate_weight{nvinfer1::DataType::kFLOAT,
                                  weight_ptr + offset_w, weight_count};
    rnn_layer->setWeightsForGate(0, trt_gate_orders[gate_index % num_gates],
                                 isW, gate_weight);
    if (num_directions == 2) {
      const float* weight_back_ptr = weight_ptr + weight_count * num_gates;
      nvinfer1::Weights gate_weight_back{
          nvinfer1::DataType::kFLOAT, weight_back_ptr + offset_w, weight_count};
      rnn_layer->setWeightsForGate(1, trt_gate_orders[gate_index % num_gates],
                                   isW, gate_weight_back);
    }
    offset_w += weight_count;
    if (gate_index % num_gates == num_gates - 1) {
      offset_w = 0;
    }
  }

  // Bias shape [dir, 4 * hidden_size] or [dir, 8 * hidden_size]
  assert(B && B->const_layer);
  bool combined_bias = (B->type.shape.dims[1] == num_gates * hidden_size);
  nvinfer1::Weights bias_w = B->const_layer->getWeights();
  g_comp->buffers.push_back(std::vector<float>(hidden_size, 0.0f));
  nvinfer1::Weights zero_bias_w{nvinfer1::DataType::kFLOAT,
                                g_comp->buffers.back().data(), hidden_size};
  const float* bias_ptr = static_cast<const float*>(bias_w.values);

  for (int gate_index = 0; gate_index < 2 * num_gates; ++gate_index) {
    bool isW = (gate_index < num_gates);
    if (!combined_bias || isW) {
      nvinfer1::Weights gate_bias{nvinfer1::DataType::kFLOAT,
                                  bias_ptr + offset_bias, hidden_size};
      rnn_layer->setBiasForGate(0, trt_gate_orders[gate_index % num_gates], isW,
                                gate_bias);
      if (num_directions == 2) {
        const float* bias_back_ptr = bias_ptr + num_gates * hidden_size;
        nvinfer1::Weights gate_bias_back{nvinfer1::DataType::kFLOAT,
                                         bias_back_ptr + offset_bias,
                                         hidden_size};
        rnn_layer->setBiasForGate(1, trt_gate_orders[gate_index % num_gates],
                                  isW, gate_bias_back);
      }
      offset_bias += hidden_size;
    } else {
      rnn_layer->setBiasForGate(0, trt_gate_orders[gate_index % num_gates], isW,
                                zero_bias_w);
      if (num_directions == 2) {
        rnn_layer->setBiasForGate(1, trt_gate_orders[gate_index % num_gates],
                                  isW, zero_bias_w);
      }
    }
  }

  // TRT result layout transformation
  // [0]: [seq, batch, dir, hidden]  --> [seq, dir, batch, hidden]
  // [1] [2] : [batch, dir, hidden]  --> [dir, batch, hidden]
  nvinfer1::ITensor* transformed_rnn[3]{rnn_layer->getOutput(0),
                                        rnn_layer->getOutput(1),
                                        rnn_layer->getOutput(2)};
  if (false && num_directions == 2) {
    for (int i = 0; i < 3; ++i) {
      auto transform_layer =
          g_comp->network->addShuffle(*rnn_layer->getOutput(i));
      if (i = 0) {
        transform_layer->setReshapeDimensions(nvinfer1::Dims{
            4, {seq_size, batch_size, num_directions, hidden_size}});
        transform_layer->setSecondTranspose(nvinfer1::Permutation{0, 2, 1, 3});
      } else {
        transform_layer->setReshapeDimensions(
            nvinfer1::Dims{3, {batch_size, num_directions, hidden_size}});
        transform_layer->setSecondTranspose(nvinfer1::Permutation{1, 0, 2});
      }
      transformed_rnn[i] = transform_layer->getOutput(0);
    }
  }
  odla_value_shape ret_shape{
      4, {seq_size, num_directions, batch_size, hidden_size}};
  odla_value_shape ret_iter_shape{3, {num_directions, batch_size, hidden_size}};
  const auto& dt = input->type.element_type;
  auto ret =
      CreateValue(transformed_rnn[0], {dt, ret_shape}, value_id.value_ids[0]);
  auto ret_h = CreateValue(transformed_rnn[1], {dt, ret_iter_shape},
                           value_id.value_ids[1]);
  auto ret_c = CreateValue(transformed_rnn[2], {dt, ret_iter_shape},
                           value_id.value_ids[2]);

  return {.size = 3, .values = {ret, ret_h, ret_c}};
}

#else
odla_values odla_LSTM(odla_value input, odla_rnn_weight_format weight_format,
                      odla_rnn_gate_order gate_order,
                      odla_value_shape weight_dims, odla_value W, odla_value R,
                      odla_value B, odla_value sequence_lens,
                      odla_value initial_h, odla_value initial_c, odla_value P,
                      odla_int32 hidden_size, odla_rnn_direction direction,
                      odla_rnn_outputs outputs, const odla_value_ids value_id) {
  //[iofc]
  const int num_gates = 4;
  const int num_directions = (direction == ODLA_RNN_BIDIRECTIONAL) ? 2 : 1;
  assert(num_direction == weight_dims.dims[0]);
  const int batch_size = input->type.shape.dims[1];
  const int len_seq = input->type.shape.dims[0];
  std::vector<nvinfer1::ActivationType> activations{
      nvinfer1::ActivationType::kSIGMOID, nvinfer1::ActivationType::kTANH,
      nvinfer1::ActivationType::kTANH};

  auto input_t = input->tensor;
  auto weight_t = W->tensor;
  auto recurrence_t = R->tensor;

  // prepare initial hidden and initial cell
  auto getInitTensor = [&num_directions, &hidden_size,
                        &batch_size](odla_value init_v) -> nvinfer1::ITensor* {
    if (init_v) {
      return init_v->tensor;
    }
    odla_value_shape dim{.size = 3,
                         .dims = {num_directions, batch_size, hidden_size}};
    g_comp->buffers.push_back(std::vector<float>(GetTotalElements(dim), 0.0f));
    nvinfer1::Weights weight{.type = nvinfer1::DataType::kFLOAT,
                             .values = g_comp->buffers.back().data(),
                             .count = GetTotalElements(dim)};
    return g_comp->network->addConstant(GetNVDims(dim), weight)->getOutput(0);
  };
  nvinfer1::ITensor* init_hidden_t = getInitTensor(initial_h);
  // LOG_VERBOSE("init_hidden dim:" +
  // gen_str(init_hidden_t->getDimensions()));
  nvinfer1::ITensor* init_cell_t = getInitTensor(initial_c);
  // LOG("init_cell dim:" << init_cell_t->getDimensions());

  assert(B && sequence_lens);
  auto bias_t = B->tensor;
  bool combined_bias = (B->type.shape.dims[1] == 4 * hidden_size);
  if (!combined_bias) {
    // Bias is of the shape of [Wb[iofc], Rb[iofc], WBb[iofc], RBb[iofc]]
    // Reshape to [[Wb[iofc], Rb[iofc]], [WBb[iofc], RBb[iofc]]]
    // in order to perform reduction to add Wb and Rb and to add WBb and RBb.
    auto reshape_bias = g_comp->network->addShuffle(*B->tensor);
    odla_value_shape dim{.size = 3,
                         .dims = {num_directions, 2, num_gates * hidden_size}};
    reshape_bias->setReshapeDimensions(GetNVDims(dim));
    bias_t = g_comp->network
                 ->addReduce(*reshape_bias->getOutput(0),
                             nvinfer1::ReduceOperation::kSUM, 2, false)
                 ->getOutput(0);
  }

  if (num_directions == 1) {
    auto squeezeDir = [](nvinfer1::ITensor* t) -> nvinfer1::ITensor* {
      auto squeeze_layer = g_comp->network->addShuffle(*t);
      squeeze_layer->setReshapeDimensions(SqueezeNVDims(t->getDimensions(), 0));
      return squeeze_layer->getOutput(0);
    };
    weight_t = squeezeDir(weight_t);
    recurrence_t = squeezeDir(recurrence_t);
    init_hidden_t = squeezeDir(init_hidden_t);
    init_cell_t = squeezeDir(init_cell_t);
  }

  // Use loops to represent recurrent layers
  nvinfer1::ILoop* rnn_loop = g_comp->network->addLoop();
  nvinfer1::ITensor* seq_lens =
      g_comp->network
          ->addConstant(
              nvinfer1::Dims{},
              nvinfer1::Weights{nvinfer1::DataType::kINT32, &len_seq, 1})
          ->getOutput(0);
  rnn_loop->addTripLimit(*seq_lens, nvinfer1::TripLimit::kCOUNT);
  // TODO: handle reverse and bidirectional
  // unsqeeze to match weight dimension
  nvinfer1::ITensor* input_iterator =
      rnn_loop->addIterator(*input_t)->getOutput(0);
  // auto unsqeeze_layer = g_comp->network->addShuffle(*input_iterator);
  // unsqeeze_layer->setName("input_iterator_unsqueeze");
  // FIXME
  // unsqeeze_layer->setReshapeDimensions(
  //  nvinfer1::Dims3{1, batch_size, input->type.shape.dims[2]});
  // input_iterator = unsqeeze_layer->getOutput(0);
  nvinfer1::IRecurrenceLayer* hidden = rnn_loop->addRecurrence(*init_hidden_t);
  nvinfer1::IRecurrenceLayer* cell = rnn_loop->addRecurrence(*init_cell_t);

  // Xt*(W^T) + Ht-1*(R^T) + (Wb + Rb)
  auto mm1 =
      g_comp->network
          ->addMatrixMultiply(*input_iterator, nvinfer1::MatrixOperation::kNONE,
                              *weight_t, nvinfer1::MatrixOperation::kTRANSPOSE)
          ->getOutput(0);
  auto mm2 = g_comp->network
                 ->addMatrixMultiply(
                     *hidden->getOutput(0), nvinfer1::MatrixOperation::kNONE,
                     *recurrence_t, nvinfer1::MatrixOperation::kTRANSPOSE)
                 ->getOutput(0);
  auto add1 =
      g_comp->network
          ->addElementWise(*mm1, *mm2, nvinfer1::ElementWiseOperation::kSUM)
          ->getOutput(0);
  auto add2 =
      g_comp->network
          ->addElementWise(*add1, *bias_t, nvinfer1::ElementWiseOperation::kSUM)
          ->getOutput(0);

  auto sliceGate = [&hidden_size, &num_directions, &batch_size](
                       nvinfer1::ITensor* gates,
                       int index) -> nvinfer1::ITensor* {
    auto slice = g_comp->network->addSlice(
        *gates, nvinfer1::Dims2{0, index * hidden_size},
        nvinfer1::Dims2{batch_size, hidden_size}, nvinfer1::Dims2{1, 1});
    return slice->getOutput(0);
  };

  auto addPeephole = [&P, &hidden_size, &num_directions](
                         nvinfer1::ITensor* gate, nvinfer1::ITensor* cell,
                         int index) -> nvinfer1::ITensor* {
    if (!P) {
      return gate;
    }
    // TODO
    auto slice = g_comp->network->addSlice(
        *P->tensor, nvinfer1::Dims2{0, index * hidden_size},
        nvinfer1::Dims2{num_directions, hidden_size}, nvinfer1::Dims2{1, 1});
    auto reshape = g_comp->network->addShuffle(*slice->getOutput(0));
    reshape->setReshapeDimensions(nvinfer1::Dims{1, {hidden_size}});

    auto mm_peep = g_comp->network
                       ->addElementWise(*reshape->getOutput(0), *cell,
                                        nvinfer1::ElementWiseOperation::kPROD)
                       ->getOutput(0);
    return g_comp->network
        ->addElementWise(*gate, *mm_peep, nvinfer1::ElementWiseOperation::kSUM)
        ->getOutput(0);
  };

  auto i_gate = sliceGate(add2, 0);
  // LOG("input gate dim:" << i_gate->getDimensions());
  // LOG_VERBOSE("input gate dim:" + gen_str(i_gate->getDimensions()));
  // it = it + P . Ct-1
  i_gate = addPeephole(i_gate, cell->getOutput(0), 0);
  // it = sigmoid(it)
  i_gate =
      g_comp->network->addActivation(*i_gate, activations.at(0))->getOutput(0);

  auto f_gate = sliceGate(add2, 2);
  // ft = ft + P . Ct-1
  f_gate = addPeephole(f_gate, cell->getOutput(0), 2);
  // ft = sigmoid(ft)
  f_gate =
      g_comp->network->addActivation(*f_gate, activations.at(0))->getOutput(0);

  auto c_gate = sliceGate(add2, 3);
  // ct = tanh(ct)
  c_gate =
      g_comp->network->addActivation(*c_gate, activations.at(1))->getOutput(0);

  // Ct = ft . Ct-1 + it . ct
  auto C_1 = g_comp->network
                 ->addElementWise(*f_gate, *cell->getOutput(0),
                                  nvinfer1::ElementWiseOperation::kPROD)
                 ->getOutput(0);
  auto C_2 = g_comp->network
                 ->addElementWise(*i_gate, *c_gate,
                                  nvinfer1::ElementWiseOperation::kPROD)
                 ->getOutput(0);
  auto C =
      g_comp->network
          ->addElementWise(*C_1, *C_2, nvinfer1::ElementWiseOperation::kSUM)
          ->getOutput(0);

  // ot
  auto o_gate = sliceGate(add2, 1);
  // ot = ot + P . Ct
  o_gate = addPeephole(o_gate, C, 1);
  // ot = sigmoid(ot)
  o_gate =
      g_comp->network->addActivation(*o_gate, activations.at(0))->getOutput(0);

  // Ht = ot . tanh(Ct)
  auto H = g_comp->network->addActivation(*C, activations.at(2))->getOutput(0);
  H = g_comp->network
          ->addElementWise(*o_gate, *H, nvinfer1::ElementWiseOperation::kPROD)
          ->getOutput(0);

  // backedge
  cell->setInput(1, *C);
  hidden->setInput(1, *H);

  // TODO: handle reverse and bidirectional
  auto output_layer =
      rnn_loop->addLoopOutput(*H, nvinfer1::LoopOutput::kCONCATENATE);
  output_layer->setInput(1, *seq_lens);
  auto hidden_out = rnn_loop->addLoopOutput(*hidden->getOutput(0),
                                            nvinfer1::LoopOutput::kLAST_VALUE);
  auto cell_out = rnn_loop->addLoopOutput(*cell->getOutput(0),
                                          nvinfer1::LoopOutput::kLAST_VALUE);

  odla_value_shape ret_shape{
      4, {len_seq, num_directions, batch_size, hidden_size}};
  odla_value_shape ret_iter_shape{3, {num_directions, batch_size, hidden_size}};

  const auto& dt = input->type.element_type;
  auto ret = CreateValue(output_layer->getOutput(0), {dt, ret_shape},
                         value_id.value_ids[0]);
  auto ret_h = CreateValue(hidden_out->getOutput(0), {dt, ret_iter_shape},
                           value_id.value_ids[1]);
  auto ret_c = CreateValue(cell_out->getOutput(0), {dt, ret_iter_shape},
                           value_id.value_ids[2]);
  return {.size = 3, .values = {ret, ret_h, ret_c}};
}
#endif

odla_value odla_HardSigmoid(odla_value input, odla_float32 alpha,
                            odla_float32 beta, const odla_value_id value_id) {
  auto layer = g_comp->network->addActivation(
      *input, nvinfer1::ActivationType::kHARD_SIGMOID);
  layer->setAlpha(alpha);
  layer->setBeta(beta);
  return CreateValue(layer->getOutput(0), input->type, value_id);
}

odla_value odla_Select(odla_value condition, odla_value a, odla_value b,
                       odla_value_shape output_dims,
                       const odla_value_id value_id) {
  nvinfer1::ITensor* lhs = a->tensor;
  nvinfer1::ITensor* rhs = b->tensor;
  nvinfer1::ITensor* cond = condition->tensor;
  const auto& dims_lhs = a->type.shape;
  const auto& dims_rhs = b->type.shape;
  const auto& dims_cond = condition->type.shape;
  if (dims_lhs.size < output_dims.size) {
    auto reshape = g_comp->network->addShuffle(*lhs);
    reshape->setReshapeDimensions(BroadcastDims(dims_lhs, output_dims.size));
    lhs = reshape->getOutput(0);
  }
  if (dims_rhs.size < output_dims.size) {
    auto reshape = g_comp->network->addShuffle(*rhs);
    reshape->setReshapeDimensions(BroadcastDims(dims_rhs, output_dims.size));
    rhs = reshape->getOutput(0);
  }
  if (dims_cond.size < output_dims.size) {
    auto reshape = g_comp->network->addShuffle(*cond);
    if (dims_cond.size == 1 && dims_cond.dims[0] == output_dims.dims[0]) {
      nvinfer1::Dims broadcast_cond_dims;
      broadcast_cond_dims.nbDims = output_dims.size;
      for (int i = 0; i < output_dims.size; ++i) {
        broadcast_cond_dims.d[i] = (i == 0) ? dims_cond.dims[0] : 1;
      }
      reshape->setReshapeDimensions(broadcast_cond_dims);
    } else {
      reshape->setReshapeDimensions(BroadcastDims(dims_cond, output_dims.size));
    }
    cond = reshape->getOutput(0);
  }
  auto select_layer = g_comp->network->addSelect(*cond, *lhs, *rhs);
  return CreateValue(select_layer->getOutput(0),
                     odla_value_type{a->type.element_type, output_dims},
                     value_id);
}

} // C extern
