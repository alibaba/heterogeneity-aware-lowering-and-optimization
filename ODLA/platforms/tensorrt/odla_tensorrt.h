//===- odla_tensorrt.h ----------------------------------------------------===//
//
// Copyright (C) 2020-2022 Alibaba Group Holding Limited.
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

#ifndef ODLA_TENSORRT_H
#define ODLA_TENSORRT_H

#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <NvInferRuntime.h>
#include <NvInferRuntimeCommon.h>
#include <ODLA/odla.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <numeric>
#include <stack>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "../include/odla_impl_common.h"
#include "ODLA/odla_common.h"
#include "common.h"
#include "plugins/initPlugin.h"

using namespace nvinfer1;

typedef struct nvmlDevice_st* nvmlDevice_t;
struct _odla_device {
  std::string name;
  std::string cuda_driver_version;
  std::string cuda_runtime_sdk_version;
  int device_count;
  int device_idx;
  std::string uuid;
  nvmlDevice_t nvml_device;
  CUdevice cu_device;
  CUcontext cu_ctx;
};

static std::string GetErrorMsg(const char* prefix, int err) {
  std::string msg(prefix);
  msg += " error: code: " + std::to_string(err);
  return msg;
}

static std::string GetErrorMsg(const char* prefix, cudaError_t err) {
  std::string msg(prefix);
  msg += std::string(" error: ") + cudaGetErrorName(err) + std::string(":") +
         cudaGetErrorString(err);
  return msg;
}

#define RETURN_ON_ERROR(lib, x, success)   \
  do {                                     \
    if (x != success) {                    \
      ODLA_LOG_ERROR(GetErrorMsg(lib, x)); \
      return ODLA_FAILURE;                 \
    }                                      \
  } while (0)

#define RETURN_ON_TRT_ERROR(call) \
  do {                            \
    if (CHECK(call) == false) {   \
      return ODLA_FAILURE;        \
    }                             \
  } while (0);

#define RETURN_ON_CUDA_ERROR(x) RETURN_ON_ERROR("cuda runtime", x, cudaSuccess)

#if !defined(ODLA_VERSION_NUMBER) || (ODLA_VERSION_NUMBER < 50)
#error This library requires minimum ODLA version 0.5
#endif

#define ODLA_TRT_MAJOR HALO_VERSION_MAJOR
#define ODLA_TRT_MINOR HALO_VERSION_MINOR
#define ODLA_TRT_PATCH HALO_VERSION_PATCH
#define ODLA_TRT_BUILD 0

// Explicitly load cuda runtime before all other ctors, so cuda rt will be
// released after calling dtors of all other global objs. This avoids the error
// of "driver shutting down".
static auto Dummy = cudaFree(0);

template <typename T>
struct TrtDestroyer {
  void operator()(T* t) {
#if NV_TENSORRT_MAJOR < 8
    t->destroy();
#else
    delete (t);
#endif
  }
};

template <typename T>
using TrtUniquePtr = std::unique_ptr<T, TrtDestroyer<T>>;

template <typename T>
std::shared_ptr<T> trt_shared_obj(T* obj) {
  return std::shared_ptr<T>(obj, TrtDestroyer<T>());
}

inline bool check(cudaError_t e, const char* file_name, int line,
                  const char* func) {
  if (e != cudaSuccess) {
    std::string msg("CUDA runtime API error ");
    msg += cudaGetErrorName(e) + std::string(":") + cudaGetErrorString(e);
    odla_GetLogger()(file_name, line, func, ODLA_LOG_LEVEL_ERROR, msg.c_str());
    return false;
  }
  return true;
}

inline bool check(bool result, const char* file_name, int line,
                  const char* func) {
  if (!result) {
    odla_GetLogger()(file_name, line, func, ODLA_LOG_LEVEL_ERROR, "trt error");
    return false;
  }
  return true;
}

#define CHECK(call) check(call, __FILE__, __LINE__, __PRETTY_FUNCTION__)

namespace open_dla_tensorrt {
class Logger : public nvinfer1::ILogger {
 public:
  void log(ILogger::Severity severity, const char* msg) NOEXCEPT override {
    odla_log_level level = ODLA_LOG_LEVEL_OFF;
    switch (severity) {
      case ILogger::Severity::kINTERNAL_ERROR:
        level = ODLA_LOG_LEVEL_ERROR;
        break;
      case ILogger::Severity::kERROR:
        level = ODLA_LOG_LEVEL_ERROR;
        break;
      case ILogger::Severity::kWARNING:
        level = ODLA_LOG_LEVEL_WARN;
        break;
      case ILogger::Severity::kINFO:
        level = ODLA_LOG_LEVEL_INFO;
        break;
      case ILogger::Severity::kVERBOSE:
        level = ODLA_LOG_LEVEL_TRACE;
        break;
      default:
        level = ODLA_LOG_LEVEL_OFF;
    }
    odla_GetLogger()("tensorrt", 0, "internal", level, msg);
  }
};
} // namespace open_dla_tensorrt

inline static void SetFP16Mode(odla_computation comp, bool value);

static open_dla_tensorrt::Logger Logger;

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
  bool is_shape_tensor = false;
};

#ifdef MAX_WORKSPACE_SIZE
static constexpr size_t MAX_WORKSPACE_SIZE_BYTES =
    (size_t)MAX_WORKSPACE_SIZE * 1024 * 1024;
#else
static constexpr size_t MAX_WORKSPACE_SIZE_BYTES = 1ul * 1024 * 1024 * 1024;
#endif
static const int MAX_INT64_CONVERTION_NUM = std::numeric_limits<int32_t>::max();

typedef struct {
  IIfConditional* branch;
  std::vector<odla_value> true_outputs;
  std::vector<odla_value> false_outputs;
  bool in_true_body;
} branch_info;

struct _odla_executable {
  odla_computation computation;
  TrtUniquePtr<nvinfer1::ICudaEngine> engine;
  _odla_executable(odla_computation computation)
      : computation(computation), engine{nullptr} {}
};

struct _odla_computation {
  odla_device device = nullptr;
  TrtUniquePtr<nvinfer1::IBuilder> builder;
  TrtUniquePtr<nvinfer1::INetworkDefinition> network;

  std::unordered_map<std::string, odla_value> inputs;
  std::unordered_map<std::string, odla_value> outputs;
  std::vector<std::vector<float>> buffers;
  std::vector<std::unique_ptr<_odla_value>> vals;
  std::vector<odla_value> input_vals;
  std::vector<odla_value> output_vals;
#if NV_TENSORRT_MAJOR >= 7
  TrtUniquePtr<nvinfer1::IBuilderConfig> builder_cfg;
  nvinfer1::IOptimizationProfile* builder_profile = nullptr;
#endif
  std::stack<branch_info> branchs;
  bool fp16_mode = false;

  bool is_dynamic_batch = false;
  int min_batch_size = 0;
  int max_batch_size = 0;
  int opt_batch_size = 0;
  bool load_engine_mode = false;
  int use_dla_core = -1;
  size_t max_workspace_size = MAX_WORKSPACE_SIZE_BYTES;

  bool is_dynamic_shape = false;
  std::unordered_map<odla_value, odla_value_shape> inputs_min_shapes;
  std::unordered_map<odla_value, odla_value_shape> inputs_max_shapes;
  std::unordered_map<odla_value, odla_value_shape> inputs_opt_shapes;

  bool is_dynamic_value = false;
  std::unordered_map<odla_value, int32_t> min_input_values;
  std::unordered_map<odla_value, int32_t> max_input_values;
  std::unordered_map<odla_value, int32_t> opt_input_values;

  _odla_executable executable;
  std::unordered_set<std::unique_ptr<_odla_context>> contexts;

  _odla_computation(odla_device dev) : device(dev), executable{this} {
    initODLAPlugin(&Logger, "");
    if (!load_engine_mode) {
      builder = TrtUniquePtr<nvinfer1::IBuilder>(
          nvinfer1::createInferBuilder(Logger));
#if NV_TENSORRT_MAJOR < 7
      builder->setMaxWorkspaceSize(MAX_WORKSPACE_SIZE_BYTES);
      network = TrtUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetwork();
#else
      nvinfer1::NetworkDefinitionCreationFlags flags = 0;
      if (const char* env_p = std::getenv("ODLA_TRT_USE_EXPLICIT_BATCH")) {
        if (*env_p != '0') {
          flags = 1U << static_cast<uint32_t>(
                      nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        }
      }
      network = TrtUniquePtr<nvinfer1::INetworkDefinition>(
          builder->createNetworkV2(flags));
      assert(network != nullptr);
      builder_cfg = TrtUniquePtr<nvinfer1::IBuilderConfig>(
          builder->createBuilderConfig());
      assert(builder_cfg != nullptr);

#endif
    }
  }

  ~_odla_computation() {
    contexts.clear(); // release all contexts before destroying executable.
    executable.engine.reset();
    network.reset();
    builder.reset();
  }

  odla_context create_context() {
    odla_SetCurrentDevice(device);
    auto context = std::make_unique<_odla_context>(this);
    auto ret = context.get();
    contexts.insert(std::move(context));
    return ret;
  }
};

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

static odla_value_shape GetOdlaShape(const nvinfer1::Dims& dims) {
  odla_value_shape ret;
  ret.size = dims.nbDims;
  for (int i = 0; i < ret.size; ++i) {
    ret.dims[i] = dims.d[i];
  }
  return ret;
}

static bool IsStaticShape(const odla_value_shape& shape) {
  if (shape.size < 0) {
    return false;
  }
  if (shape.size == 0) {
    return shape.dims[0] >= 0;
  }
  for (int i = 0; i < shape.size; ++i) {
    if (shape.dims[i] < 0) {
      return false;
    }
  }
  return true;
}

struct _odla_context {
  odla_computation comp = nullptr;
  CUcontext cu_ctx = nullptr;
  TrtUniquePtr<nvinfer1::IExecutionContext> ctx{nullptr};
  std::vector<void*> bindings;

  cudaStream_t stream = nullptr;
  cudaGraph_t graph;
  cudaGraphExec_t instance;
  bool enable_cuda_graph = false;
  bool is_captured = false;

  typedef struct {
    void* host_ptr = nullptr;
    void* dev_ptr = nullptr;
    size_t total_size = 0; // max memory size.
    size_t used_size = 0;  // used memory size.
    // odla_value_type vt;
  } OutputPtrInfo;

  typedef struct {
    void* dev_ptr = nullptr;
    size_t total_size = 0;
  } InputPtrInfo;
  std::unordered_map<odla_value, OutputPtrInfo> output_ptrs;
  std::unordered_map<odla_value, InputPtrInfo> input_ptrs;

  int run_batch_size = 0;

  std::unordered_map<odla_value, odla_value_shape> real_shapes;

  _odla_context(odla_computation comp) : comp(comp) {
    for (auto& v : comp->input_vals) {
      real_shapes[v] = v->type.shape;
    }
    if (comp->executable.engine == nullptr) {
      odla_executable exec;
      odla_CompileComputation(comp, nullptr, &exec);
    }
    cuDevicePrimaryCtxRetain(&cu_ctx, comp->device->cu_device);
    ctx = TrtUniquePtr<IExecutionContext>(
        comp->executable.engine->createExecutionContext());
    assert(ctx != nullptr);
    CHECK(cudaStreamCreate(&stream));
  }

  ~_odla_context() {
    for (const auto& ptr_info : input_ptrs) {
      cudaFree(ptr_info.second.dev_ptr);
    }
    for (const auto& ptr_info : output_ptrs) {
      cudaFree(ptr_info.second.dev_ptr);
    }
    CHECK(cudaStreamDestroy(stream));
    stream = nullptr;
    comp = nullptr;
  }
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

void SetFP16Mode(odla_computation comp, bool value) {
#if NV_TENSORRT_MAJOR >= 8
  if (value) {
    comp->builder_cfg->setFlag(BuilderFlag::kFP16);
  } else {
    comp->builder_cfg->clearFlag(BuilderFlag::kFP16);
  }
#else
  comp->builder->setFp16Mode(value);
#endif
}

static odla_device GetDefaultDevice() {
  static std::unique_ptr<_odla_device> dev;
  if (dev == nullptr) {
    odla_device d;
    odla_AllocateDevice(0, ODLA_DEVICE_NVIDIA_GPU, 0, &d);
    dev.reset(d);
  }
  return dev.get();
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

static std::string GetName(odla_value_id id, const char* suffix = nullptr) {
  return std::string(reinterpret_cast<const char*>(id)) + suffix;
}

static std::string GetName(const odla_value& value, const char* suffix) {
  return GetName((odla_value_id)(value->name), suffix);
}

static odla_value_type ValidateValueType(const odla_value_type& type) {
  // Trt doesn't support INT64, convert value_type of ODLA_INT64 to ODLA_INT32
  if (type.element_type == ODLA_INT64) {
    return odla_value_type{ODLA_INT32, type.shape};
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

#endif // ODLA_TENSORRT_H
