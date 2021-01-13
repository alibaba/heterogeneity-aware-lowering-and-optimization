//===- odla_tensorrt.cc ---------------------------------------------------===//
//
// Copyright (C) 2019-2020 Alibaba Group Holding Limited.
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
//#include "plugin/nmsPlugin/nmsPlugin.h"
#include <ODLA/odla.h>
#include <bits/stdint-intn.h>
#include <cuda_runtime.h>

#include <cassert>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <memory>
#include <numeric>
#include <unordered_map>
#include <vector>

using namespace nvinfer1;

#if !defined(ODLA_VERSION_NUMBER) || (ODLA_VERSION_NUMBER < 50)
#error This library requires minimum ODLA version 0.5
#endif

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
  void log(ILogger::Severity severity, const char* msg) override {
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

struct _odla_value {
  _odla_value(nvinfer1::ITensor* tensor, const odla_value_type& type,
              const char* name)
      : tensor(tensor), type(type) {
    tensor->setName(name);
  }
  _odla_value(nvinfer1::ILayer* layer, const odla_value_type& type,
              const char* name)
      : layer(layer), tensor(layer->getOutput(0)), type(type) {
    layer->setName(name);
  }
  operator nvinfer1::ITensor&() { return *tensor; }
  nvinfer1::ILayer* layer = nullptr;
  nvinfer1::ITensor* tensor = nullptr;
  nvinfer1::IConstantLayer* const_layer = nullptr;
  odla_value_type type;
};

#ifdef MAX_WORKSPACE_SIZE
static constexpr size_t MAX_WORKSPACE_SIZE_BYTES =
    (size_t)MAX_WORKSPACE_SIZE * 1024 * 1024;
#else
static constexpr size_t MAX_WORKSPACE_SIZE_BYTES = 1ul * 1024 * 1024 * 1024;
#endif
static const int MAX_INT64_CONVERTION_NUM = 65536ul;

struct _odla_computation {
  nvinfer1::IBuilder* builder;
  nvinfer1::INetworkDefinition* network;
  std::unordered_map<std::string, odla_value> inputs;
  std::unordered_map<std::string, odla_value> outputs;
  std::vector<std::vector<float>> buffers;
  std::vector<std::unique_ptr<_odla_value>> vals;
  bool fp16_mode = false;

  bool is_dynamic_batch = false;
  int min_batch_size = 0;
  int max_batch_size = 0;
  int opt_batch_size = 0;
  size_t max_workspace_size = MAX_WORKSPACE_SIZE_BYTES;

  _odla_computation() {
    builder = nvinfer1::createInferBuilder(Logger);
    if (const char* env_p = std::getenv("ODLA_TRT_MAX_WS_MB")) {
      if (int mb = std::stoi(env_p); mb != 0) {
        max_workspace_size = mb << 20;
      }
    }

#if NV_TENSORRT_MAJOR < 7
    builder->setMaxWorkspaceSize(max_workspace_size);
    network = builder->createNetwork();
#else
#ifdef USE_PLUGIN
    initLibNvInferPlugins(static_cast<void*>(&Logger), "");
#endif
    nvinfer1::NetworkDefinitionCreationFlags flags = 0;
    network = builder->createNetworkV2(flags);
#endif
  }

  ~_odla_computation() {
    builder->destroy();
    network->destroy();
  }
};

struct _odla_context {
  odla_computation comp;
  nvinfer1::ICudaEngine* engine = nullptr;
  nvinfer1::IExecutionContext* ctx = nullptr;
#if NV_TENSORRT_MAJOR >= 7
  nvinfer1::IBuilderConfig* builder_cfg = nullptr;
  nvinfer1::IOptimizationProfile* builder_profile = nullptr;

#endif

  typedef struct {
    void* host_ptr;
    void* dev_ptr;
    size_t len;
    odla_value_type vt;
  } OutputPtrInfo;

  typedef struct {
    const void* host_ptr;
    void* dev_ptr;
  } InputPtrInfo;
  std::unordered_map<std::string, OutputPtrInfo> output_ptrs;
  std::unordered_map<std::string, InputPtrInfo> input_ptrs;
  int run_batch_size = 0;
  _odla_context(odla_computation comp) : comp(comp) {
#if NV_TENSORRT_MAJOR < 7
    engine = comp->builder->buildCudaEngine(*comp->network);
#else
    builder_cfg = comp->builder->createBuilderConfig();

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
    engine = comp->builder->buildEngineWithConfig(*comp->network, *builder_cfg);
#endif
    ctx = engine->createExecutionContext();
  }
  ~_odla_context() {
    engine->destroy();
    ctx->destroy();
#if NV_TENSORRT_MAJOR >= 7
    builder_cfg->destroy();
#endif
  }
};

static int64_t GetTotalElements(const odla_value_shape& dims) {
  return std::accumulate(dims.dims, dims.dims + dims.size, 1,
                         std::multiplies<size_t>());
}

static nvinfer1::Dims GetNVDims(int n, const odla_uint32* dims) {
  nvinfer1::Dims ret;
  assert(n <= nvinfer1::Dims::MAX_DIMS);
  ret.nbDims = n;
  for (int i = 0; i < n; ++i) {
    ret.d[i] = static_cast<int>(dims[i]);
  }
  return ret;
}

static nvinfer1::Dims GetNVDims(const odla_value_shape& dims) {
  nvinfer1::Dims ret;
  ret.nbDims = dims.size;
  for (int i = 0, e = std::min(nvinfer1::Dims::MAX_DIMS, ODLA_MAX_DIMENSION);
       i < e; ++i) {
    ret.d[i] = dims.dims[i];
  }
  return ret;
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

thread_local odla_computation g_comp;
static std::vector<std::unique_ptr<_odla_computation>> g_comps;
static std::vector<int> g_workspace;

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

static void* ValidateValuePtr(const odla_value_type& type, void* ptr) {
  if (type.element_type == ODLA_INT64) {
    int64_t* src = static_cast<int64_t*>(ptr);
    auto num_elements = GetTotalElements(type.shape);
    auto workspace_size = g_workspace.size();
    assert(workspace_size + num_elements < MAX_INT64_CONVERTION_NUM);
    int* tmp = g_workspace.data() + workspace_size;
    for (int i = 0; i < num_elements; ++i) {
      g_workspace.push_back(static_cast<int>(*src++));
    }
    return tmp;
  }
  return ptr;
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

extern "C" {
odla_status odla_CreateComputation(odla_computation* computation) {
  g_comps.push_back(std::make_unique<_odla_computation>());
  g_comp = g_comps.back().get();
  *computation = g_comp;
  g_workspace.reserve(MAX_INT64_CONVERTION_NUM);
  return ODLA_SUCCESS;
}

odla_status odla_SetActiveComputation(odla_computation computation) {
  g_comp = computation;
  return ODLA_SUCCESS;
}

odla_status odla_DestroyComputation(odla_computation comp) {
  for (auto& c : g_comps) {
    if (c.get() == comp) {
      c.reset();
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
        computation->network->destroy();
        nvinfer1::NetworkDefinitionCreationFlags flags =
            1U << static_cast<uint32_t>(
                nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        computation->network = computation->builder->createNetworkV2(flags);
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
  return v;
}

odla_value odla_CreateConstant(odla_value_type type, const void* ptr,
                               const odla_value_id id) {
  nvinfer1::Weights weight{
      .type = GetNVDataType(type.element_type),
      .values = ValidateValuePtr(type, const_cast<void*>(ptr)),
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
  val->tensor->setName(name);
  g_comp->network->markOutput(*val->tensor);
  return ODLA_SUCCESS;
}

odla_status odla_BindToArgument(odla_value value, const odla_void* data_ptr,
                                odla_context context) {
  void* dev_ptr = nullptr;
  odla_value_shape real_shape = value->type.shape;
  if (g_comp && g_comp->is_dynamic_batch) {
    real_shape.dims[0] = context->run_batch_size;
  }
  size_t bytes =
      GetTotalElements(real_shape) * GetElementSize(value->type.element_type);
  CHECK(cudaMalloc(&dev_ptr, bytes));
  void* validated_data_ptr =
      ValidateValuePtr(value->type, const_cast<void*>(data_ptr));
  CHECK(cudaMemcpy(dev_ptr, validated_data_ptr, bytes, cudaMemcpyHostToDevice));
  context->input_ptrs[value->tensor->getName()] = {.host_ptr = data_ptr,
                                                   .dev_ptr = dev_ptr};
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
  void* dst = nullptr;
  odla_value_shape real_shape = value->type.shape;
  if (g_comp && g_comp->is_dynamic_batch) {
    real_shape.dims[0] = context->run_batch_size;
  }
  size_t bytes =
      GetTotalElements(real_shape) * GetElementSize(value->type.element_type);

  CHECK(cudaMalloc(&dst, bytes));
  context->output_ptrs[value->tensor->getName()] = {
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

  return CreateValue(sub, {lhs->type.element_type, out_dim}, id);
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
  auto op0 = g_comp->network->addUnary(*input, nvinfer1::UnaryOperation::kSINH);
  auto op1 = g_comp->network->addUnary(*input, nvinfer1::UnaryOperation::kCOSH);
  auto op = g_comp->network->addElementWise(
      *(op0->getOutput(0)), *(op1->getOutput(0)),
      nvinfer1::ElementWiseOperation::kDIV);
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
  assert(input_layout == odla_memory_layout::ODLA_CHANNELS_FIRST);
  assert(type == ODLA_FLOAT32);

  int64_t C = input_dims.dims[1];

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
  return CreateValue(bn, input->type, value_id);
}

#ifdef USE_PLUGIN
odla_value odla_InstanceNormalization(
    odla_value input, odla_memory_layout input_layout, odla_value mean,
    odla_value var, odla_float32 epsilon, odla_value scale, odla_value offset,
    odla_float32 scalar_scale, odla_float32 scalar_offset,
    const odla_value_id value_id) {
  std::vector<nvinfer1::ITensor*> inputs = {input->tensor, scale->tensor,
                                            offset->tensor};
  const static char* plugin_name = "InstanceNormalization_TRT";
  const static char* plugin_ver = "1";
  auto creator = getPluginRegistry()->getPluginCreator(plugin_name, plugin_ver);
  std::vector<nvinfer1::PluginField> f;
  f.emplace_back("epsilon", &epsilon, nvinfer1::PluginFieldType::kFLOAT32, 1);
  nvinfer1::PluginFieldCollection plugin_data;
  plugin_data.nbFields = f.size();
  plugin_data.fields = f.data();
  auto plugin = creator->createPlugin(plugin_name, &plugin_data);
  auto norm = g_comp->network->addPluginV2(
      &inputs[0], static_cast<int>(inputs.size()), *plugin);
  return CreateValue(norm->getOutput(0), input->type, value_id);
}
#endif

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
  unsigned reduce_axes = axis < 0 ? input->type.shape.size - 1 : axis;
  auto topk = g_comp->network->addTopK(*input, nvinfer1::TopKOperation::kMAX, 1,
                                       1 << reduce_axes);
  return CreateValue(topk->getOutput(1), output_value_type, id);
}

odla_value odla_Gather(odla_value input, const odla_value indices,
                       odla_int32 axis, odla_value_shape output_dims,
                       const odla_value_id id) {
  axis = axis < 0 ? input->type.shape.size - 1 : axis;
  auto gather = g_comp->network->addGather(*input, *indices, axis);
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
  auto creator = getPluginRegistry()->getPluginCreator("BatchedNMS_TRT", "1");
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

} // C extern
