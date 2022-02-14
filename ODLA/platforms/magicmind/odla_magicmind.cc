//===- odla_magicmind.cc --------------------------------------------------===//
//
// Copyright (C) 2019-2021 Alibaba Group Holding Limited.
// Copyright (C) [2022] by Cambricon, Inc. All rights reserved.
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

#include <ODLA/odla.h>
#include <stdarg.h>
#include <stdio.h>

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <stack>
#include <string>
#include <unordered_map>
#include <vector>

#include "calibration.h"
#include "cnrt.h"
#include "interface_builder.h"
#include "interface_network.h"
#include "interface_runtime.h"
#include "utils.h"

#define USE_CNRT_NOTIFIER 1
#define USE_CLOCK_TIME 1
#define USE_INT8_QUANT 0

#define DUMP_DATA 0

#define USE_UNIQUE_PTR 0

#define DEBUG 0
#if DEBUG
#define LOG_PRINT(fmt, args...) printf(fmt, ##args)
#else
#define LOG_PRINT(fmt, args...)
#endif

#if !defined(ODLA_VERSION_NUMBER) || (ODLA_VERSION_NUMBER < 50)
#error This library requires minimum ODLA version 0.5
#endif

#define CHECK(status)                                  \
  do {                                                 \
    auto ret = (status);                               \
    if (ret != magicmind::Status::OK()) {              \
      std::cout << "mm failure: " << ret << std::endl; \
      abort();                                         \
    }                                                  \
  } while (0)

#define LOGINFO(fmt, args...) fprintf(stdout, "[MMINFO]  " fmt "\n", ##args)

#define PTR_CHECK(ptr)                         \
  do {                                         \
    if (ptr == nullptr) {                      \
      std::cout << "mm failure " << std::endl; \
      abort();                                 \
    }                                          \
  } while (0)

struct _odla_value {
  _odla_value(magicmind::ITensor* tensor, const odla_value_type& type,
              const char* name)
      : tensor(tensor), type(type) {
    tensor->SetTensorName(name);
    real_name = name;
  }
  _odla_value(magicmind::INode* layer, const odla_value_type& type,
              const char* name)
      : layer(layer), tensor(layer->GetOutput(0)), type(type) {
    layer->SetNodeName(name);
    tensor->SetTensorName(name);
    real_name = name;
  }
  operator magicmind::ITensor&() { return *tensor; }
  magicmind::INode* layer = nullptr;
  magicmind::ITensor* tensor = nullptr;
  magicmind::IConstNode* const_layer = nullptr;
  std::string real_name;
  odla_value_type type;
};

typedef struct {
  magicmind::IIfNode* branch;
  magicmind::ICondBody* then_body;
  magicmind::ICondBody* else_body;
  std::vector<odla_value> true_outputs;
  std::vector<odla_value> false_outputs;
  bool in_true_body;
} branch_info;

static const int MAX_INT64_CONVERTION_NUM = 65536ul;
struct _odla_computation {
  magicmind::IBuilder* builder;
  magicmind::INetwork* network;
  magicmind::INetwork* network_backup;

  std::vector<magicmind::ITensor*> input_tensors;
  std::vector<magicmind::ITensor*> output_tensors;
  std::unordered_map<std::string, odla_value> inputs;
  std::unordered_map<std::string, odla_value> outputs;
  std::vector<std::string> inputs_names;
  std::vector<std::vector<float>> buffers;
  std::vector<std::unique_ptr<_odla_value>> vals;
  std::stack<branch_info> branchs;
  bool fp16_mode = false;

  bool is_dynamic_batch = false;
  int min_batch_size = 0;
  int max_batch_size = 0;
  int opt_batch_size = 0;

  _odla_computation() {
    cnrtDev_t dev;
    cnrtInit(0);
    cnrtGetDeviceHandle(&dev, 0);
    cnrtSetCurrentDevice(dev);

    builder = magicmind::CreateIBuilder();
    network = magicmind::CreateINetwork();
    network_backup = network;
  }

  ~_odla_computation() {
    builder->Destroy();
    network->Destroy();
    LOG_PRINT("odla_computation destroy\n");
  }
};

thread_local odla_computation g_comp;
static std::vector<std::unique_ptr<_odla_computation>> g_comps;

struct _odla_context {
  odla_computation comp;
  magicmind::IModel* model = nullptr;
  magicmind::IEngine* engine = nullptr;
  magicmind::IContext* ctx = nullptr;
  magicmind::IBuilderConfig* builder_cfg = nullptr;
  std::vector<magicmind::IRTTensor*> irt_input_tensors;
  std::vector<magicmind::IRTTensor*> irt_output_tensors;
#if USE_CNRT_NOTIFIER
  cnrtNotifier_t notifier_start;
  cnrtNotifier_t notifier_end;
#endif
  cnrtQueue_t queue;

  typedef struct {
    void* host_ptr;
    void* dev_ptr;
    size_t len;
    odla_value_type vt;
  } OutputPtrInfo;

  typedef struct {
    const void* host_ptr;
    void* dev_ptr;
    size_t len;
  } InputPtrInfo;
  std::unordered_map<std::string, OutputPtrInfo> output_ptrs;
  std::unordered_map<std::string, InputPtrInfo> input_ptrs;
  int run_batch_size = 0;
  _odla_context(odla_computation comp) : comp(comp) {
    builder_cfg = magicmind::CreateIBuilderConfig();
    std::vector<std::string> archs{"mtp_372"};
    CHECK(builder_cfg->SetMLUArch(archs));
    std::string cfg_string = "";
    cfg_string += R"({
       )";
    if (comp->fp16_mode) {
      cfg_string += R"("precision_config": {
        "precision_mode": "force_float16"
        })";
    } /*else {
        cfg_string += R"("precision_config": {
        "precision_mode": "force_float32"
        },)";
    }*/
    if (comp->is_dynamic_batch) {
      if (comp->fp16_mode) {
        cfg_string += R"(,)";
      }
      cfg_string += R"(
          "graph_shape_mutable": true,
          "dim_range": {
            )";
      for (int i = 0; i < comp->inputs_names.size(); i++) {
        if (i != 0) {
          cfg_string += R"(,
                )";
        }
        cfg_string += R"(")" + std::to_string(i) + R"(": {
              "min":[)";
        auto input_value = comp->inputs[comp->inputs_names[i]];
        for (int j = 0; j < input_value->type.shape.size; j++) {
          if (input_value->type.shape.dims[j] < 0) {
            cfg_string += std::to_string(comp->min_batch_size);
          } else {
            cfg_string += std::to_string(input_value->type.shape.dims[j]);
          }
          if (j != input_value->type.shape.size - 1) cfg_string += ", ";
        }
        cfg_string += R"(],
              "max": [)";
        for (int j = 0; j < input_value->type.shape.size; j++) {
          if (input_value->type.shape.dims[j] < 0) {
            cfg_string += std::to_string(comp->max_batch_size);
          } else {
            cfg_string += std::to_string(input_value->type.shape.dims[j]);
          }
          if (j != input_value->type.shape.size - 1) cfg_string += ", ";
        }
        cfg_string += R"(]
            })";
      }
      cfg_string += R"(
        })";
    }
    cfg_string += R"(
    })";
    LOG_PRINT("\n%s\n", cfg_string.c_str());
    if (!cfg_string.empty()) {
      builder_cfg->ParseFromString(cfg_string);
    }

#if USE_INT8_QUANT
    std::vector<std::string> data_paths;
    data_paths.push_back("xxx/ILSVRC2012_val_00000001");
    data_paths.push_back("xxx/ILSVRC2012_val_00000002");
    data_paths.push_back("xxx/ILSVRC2012_val_00000003");
    data_paths.push_back("xxx/ILSVRC2012_val_00000004");
    comp->network->SetPrecision(magicmind::PrecisionMode::INT8_MIXED_FLOAT32);
    magicmind::Dims op_dims({1, 3, 224, 224});
    FixedCalibData calib_data(op_dims, magicmind::DataType::FLOAT32,
                              data_paths.size(), data_paths);
    // perform calibration on network
    auto algorithm =
        magicmind::QuantizationAlgorithm::SYMMETRIC_LINEAR_ALGORITHM;
    std::unique_ptr<magicmind::ICalibrator> calibrator(
        magicmind::CreateICalibrator({&calib_data}));
    calibrator->SetQuantizationAlgorithm(algorithm);
    calibrator->Calibrate(comp->network);
#endif
    model = comp->builder->BuildModel("magicmind_model", comp->network,
                                      builder_cfg);
    //    model->SerializeToFile("mm-odla-model");
    engine = model->CreateIEngine();
    ctx = engine->CreateIContext();
    LOG_PRINT("create context\n");
    CHECK(CreateInputTensors(ctx, &irt_input_tensors));
    if (!comp->is_dynamic_batch) {
      CHECK(CreateOutputTensors(ctx, &irt_output_tensors));
    }
#if DUMP_DATA
    magicmind::IContext::ContextDumpInfo dump_info;
    dump_info.dump_mode = 2;
    dump_info.path = "./dump";
    dump_info.file_format = 1;
    ctx->SetContextDumpInfo(dump_info);
#endif
#if USE_CNRT_NOTIFIER
    cnrtCreateNotifier(&notifier_start);
    cnrtCreateNotifier(&notifier_end);
#endif
    cnrtQueueCreate(&queue);
  }
  ~_odla_context() {
#if USE_CNRT_NOTIFIER
    cnrtDestroyNotifier(&notifier_start);
    cnrtDestroyNotifier(&notifier_end);
#endif
    cnrtQueueDestroy(queue);

    LOG_PRINT("start to destroy context\n");
    ctx->Destroy();
    engine->Destroy();
    model->Destroy();
    builder_cfg->Destroy();
    // copy results and free temp buffers.
    for (auto& kv : input_ptrs) {
      CNRT_CHECK(cnrtFree(kv.second.dev_ptr));
      LOG_PRINT("input free %p\n", kv.second.dev_ptr);
    }
    for (auto& kv : output_ptrs) {
      if (!g_comp->is_dynamic_batch) {
        CNRT_CHECK(cnrtFree(kv.second.dev_ptr));
        LOG_PRINT("output free %p\n", kv.second.dev_ptr);
      }
    }
    input_ptrs.clear();
    output_ptrs.clear();
  }
};

thread_local odla_context g_ctx;
#if USE_UNIQUE_PTR
static std::vector<std::unique_ptr<_odla_context>> g_ctxs;
#endif
static std::vector<int> g_workspace;

static int64_t GetTotalElements(const odla_value_shape& dims) {
  return std::accumulate(dims.dims, dims.dims + dims.size, 1,
                         std::multiplies<size_t>());
}
static magicmind::Dims GetMMDims(int n, const odla_uint32* dims) {
  std::vector<int64_t> ret_dims;
  for (int i = 0; i < n; ++i) {
    ret_dims.push_back(static_cast<int64_t>(dims[i]));
  }
  return magicmind::Dims(ret_dims);
}

static magicmind::Dims GetMMDims(const odla_value_shape& dims) {
  std::vector<int64_t> ret_dims;
  int e = std::min(dims.size, ODLA_MAX_DIMENSION);
  for (int i = 0; i < e; ++i) {
    ret_dims.push_back(static_cast<int64_t>(dims.dims[i]));
  }
  return magicmind::Dims(ret_dims);
}

static odla_value_shape BroadcastDims(const odla_value_shape& dims,
                                      size_t dim_size) {
  if (dims.size >= dim_size) {
    return dims;
  }
  odla_value_shape ret_shape;
  ret_shape.size = dim_size;
  for (int i = 0, e = dim_size - dims.size; i != e; ++i) {
    ret_shape.dims[i] = 1;
  }
  for (int i = dim_size - dims.size, j = 0; i != dim_size; ++i, ++j) {
    ret_shape.dims[i] = dims.dims[j];
  }
  return ret_shape;
}

static magicmind::Layout GetMMLayout(odla_memory_layout input_layout) {
  switch (input_layout) {
    case ODLA_CHANNELS_FIRST:
    case ODLA_OIS:
      return magicmind::Layout::NCHW;
    case ODLA_CHANNELS_LAST:
      return magicmind::Layout::NHWC;
    case ODLA_SIO:
      return magicmind::Layout::HWCN;
    default:
      return magicmind::Layout::NONE;
  }
}

static magicmind::DataType GetMMDataType(odla_element_type type) {
  switch (type) {
    case ODLA_FLOAT32:
      return magicmind::DataType::FLOAT32;
    case ODLA_FLOAT16:
      return magicmind::DataType::FLOAT16;
    case ODLA_INT32:
    case ODLA_INT64:
      return magicmind::DataType::INT32;
    case ODLA_INT8:
      return magicmind::DataType::INT8;
    case ODLA_BOOL:
      return magicmind::DataType::BOOL;
    default:
      return magicmind::DataType::FLOAT32;
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
  delete g_ctx;
  //  for (auto& c : g_comps) {
  //    if (c.get() == comp) {
  //      c.reset();
  //      return ODLA_SUCCESS;
  //    }
  //  }
  //  assert(0);
  return ODLA_FAILURE;
}

odla_status odla_SetComputationItem(odla_computation computation,
                                    odla_item_type type,
                                    odla_item_value value) {
  switch (type) {
    case ODLA_DYNAMIC_BATCH:
      computation->is_dynamic_batch = *(reinterpret_cast<bool*>(value));
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

    case ODLA_BF16_MODE:
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
#if USE_UNIQUE_PTR
  g_ctxs.push_back(std::make_unique<_odla_context>(g_comp));
  g_ctx = g_ctxs.back().get();
#else
  g_ctx = new _odla_context(g_comp);
#endif
  *context = g_ctx;
  return ODLA_SUCCESS;
}

odla_status odla_DestroyContext(odla_context context) {
  assert(0);
  return ODLA_FAILURE;
}

odla_value odla_CreateArgument(odla_value_type type, const odla_value_id id) {
  const char* name = reinterpret_cast<const char*>(id);
  auto input = g_comp->network->AddInput(GetMMDataType(type.element_type),
                                         GetMMDims(type.shape));
  odla_value v = CreateValue(input, type, id);
  g_comp->inputs[name] = v;
  g_comp->inputs_names.push_back(name);
  g_comp->input_tensors.push_back(input);
  return v;
}

odla_value odla_CreateConstant(odla_value_type type, const void* ptr,
                               const odla_value_id id) {
  auto c = g_comp->network->AddIConstNode(
      GetMMDataType(type.element_type), GetMMDims(type.shape),
      ValidateValuePtr(type, const_cast<void*>(ptr)));
  odla_value v = CreateValue(c->GetOutput(0), ValidateValueType(type), id);
  v->const_layer = c;
  return v;
}

odla_status odla_SetValueAsOutput(const odla_value val) {
  if (!g_comp->branchs.empty()) {
    auto& br_info = g_comp->branchs.top();
    if (br_info.in_true_body) {
      br_info.true_outputs.push_back(val);
    } else {
      br_info.false_outputs.push_back(val);
    }
    return ODLA_SUCCESS;
  }

  g_comp->outputs[val->real_name.c_str()] = val;
  g_comp->output_tensors.push_back(val->tensor);
  g_comp->network->MarkOutput(val->tensor);
  return ODLA_SUCCESS;
}

odla_status odla_BindToArgument(odla_value value, const odla_void* data_ptr,
                                odla_context context) {
  void* dev_ptr = nullptr;
  odla_value_shape real_shape = value->type.shape;
  if ((g_comp && g_comp->is_dynamic_batch) || context->run_batch_size) {
    real_shape.dims[0] = context->run_batch_size;
  }
  size_t bytes =
      GetTotalElements(real_shape) * GetElementSize(value->type.element_type);
  auto iter = context->input_ptrs.find(value->tensor->GetTensorName());
  if ((iter == context->input_ptrs.end()) || (iter->second.len < bytes)) {
    if ((iter != context->input_ptrs.end()) && (iter->second.len < bytes)) {
      cnrtFree(iter->second.dev_ptr);
    }
    CNRT_CHECK(cnrtMalloc(&dev_ptr, bytes));
    context->input_ptrs[value->tensor->GetTensorName()] = {
        .host_ptr = data_ptr, .dev_ptr = dev_ptr, .len = bytes};
    LOG_PRINT("input alloc %p\n", dev_ptr);
  } else {
    iter->second.host_ptr = data_ptr;
    LOG_PRINT("input reuse %p\n", iter->second.dev_ptr);
  }
  void* validated_data_ptr =
      ValidateValuePtr(value->type, const_cast<void*>(data_ptr));
  CNRT_CHECK(
      cnrtMemcpy(context->input_ptrs[value->tensor->GetTensorName()].dev_ptr,
                 validated_data_ptr, bytes, CNRT_MEM_TRANS_DIR_HOST2DEV));
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
  void* dev_ptr = nullptr;
  odla_value_shape real_shape = value->type.shape;
  if ((g_comp && g_comp->is_dynamic_batch) || context->run_batch_size) {
    real_shape.dims[0] = context->run_batch_size;
  }
  size_t bytes =
      GetTotalElements(real_shape) * GetElementSize(value->type.element_type);
  if (g_comp->is_dynamic_batch) {
    context->output_ptrs[value->tensor->GetTensorName()] = {
        .host_ptr = data_ptr,
        .dev_ptr = nullptr,
        .len = bytes,
        .vt = value->type};
    return ODLA_SUCCESS;
  }
  auto iter = context->output_ptrs.find(value->tensor->GetTensorName());
  if ((iter == context->output_ptrs.end()) || (iter->second.len < bytes)) {
    if ((iter != context->output_ptrs.end()) && (iter->second.len < bytes)) {
      cnrtFree(iter->second.dev_ptr);
    }
    CNRT_CHECK(cnrtMalloc(&dev_ptr, bytes));
    context->output_ptrs[value->tensor->GetTensorName()] = {
        .host_ptr = data_ptr,
        .dev_ptr = dev_ptr,
        .len = bytes,
        .vt = value->type};
    LOG_PRINT("output alloc %p\n", dev_ptr);
  } else {
    iter->second.host_ptr = data_ptr;
    LOG_PRINT("output reuse %p\n", iter->second.dev_ptr);
  }

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
#if USE_CNRT_NOTIFIER
  cnrtPlaceNotifier(context->notifier_start, context->queue);
#endif
#if USE_CLOCK_TIME
  uint64_t time_start = EnvTime::NowNanos();
#endif
  for (int i = 0; i < comp->input_tensors.size(); i++) {
    auto input_value = comp->inputs[comp->inputs_names[i]];
    odla_value_shape real_shape = input_value->type.shape;
    if (real_shape.dims[0] < 0) {
      real_shape.dims[0] = context->run_batch_size;
    }
    CHECK(context->irt_input_tensors[i]->SetDimensions(GetMMDims(real_shape)));
    // CHECK(irt_input_tensors[i]->SetDimensions(op_dims));
    auto dev_ptr =
        context->input_ptrs[comp->input_tensors[i]->GetTensorName()].dev_ptr;
    CHECK(context->irt_input_tensors[i]->SetData(dev_ptr));
  }
  if (!comp->is_dynamic_batch) {
    CHECK(context->ctx->InferOutputShape(context->irt_input_tensors,
                                         context->irt_output_tensors));
    for (int i = 0; i < comp->output_tensors.size(); i++) {
      auto dev_ptr =
          context->output_ptrs[comp->output_tensors[i]->GetTensorName()]
              .dev_ptr;
      CHECK(context->irt_output_tensors[i]->SetData(dev_ptr));
    }
  }
  if (!comp->is_dynamic_batch) {
    CHECK(context->ctx->Enqueue(context->irt_input_tensors,
                                context->irt_output_tensors, context->queue));
  } else {
    CHECK(context->ctx->Enqueue(context->irt_input_tensors,
                                &context->irt_output_tensors, context->queue));
  }
#if USE_CNRT_NOTIFIER
  cnrtPlaceNotifier(context->notifier_end, context->queue);
#endif
  CNRT_CHECK(cnrtQueueSync(context->queue));
#if USE_CLOCK_TIME
  uint64_t time_end = EnvTime::NowNanos();
  uint64_t dur = (time_end - time_start) / 1000;
  printf("ExecuteComputation time is %lu us\n", dur);
#endif
#if USE_CNRT_NOTIFIER
  cnrtWaitNotifier(context->notifier_start);
  cnrtWaitNotifier(context->notifier_end);
  float ptv = 0.0f;
  cnrtNotifierElapsedTime(context->notifier_start, context->notifier_end, &ptv);
  std::cout << "Notifier time : " << ptv << " ms" << std::endl;
#endif
  for (int i = 0; i < comp->output_tensors.size(); i++) {
    void* output_dev_ptr = nullptr;
    if (!comp->is_dynamic_batch) {
      output_dev_ptr =
          context->output_ptrs[comp->output_tensors[i]->GetTensorName()]
              .dev_ptr;
    } else {
      output_dev_ptr = context->irt_output_tensors[i]->GetMutableData();
    }
    auto output_host_ptr =
        context->output_ptrs[comp->output_tensors[i]->GetTensorName()].host_ptr;
    LOG_PRINT("copy output mlu %p size %lu\n", output_dev_ptr,
              context->irt_output_tensors[i]->GetSize());
    CNRT_CHECK(cnrtMemcpy(output_host_ptr, output_dev_ptr,
                          context->irt_output_tensors[i]->GetSize(),
                          CNRT_MEM_TRANS_DIR_DEV2HOST));
  }

  if (comp->is_dynamic_batch) {
    for (auto tensor : context->irt_output_tensors) tensor->Destroy();
    context->irt_output_tensors.clear();
  }
  // to-do int64-int32 convert
  return ODLA_SUCCESS;
}

static odla_value_shape GetBroadcastOutputShape(odla_value_shape dims_lhs,
                                                odla_value_shape dims_rhs) {
  odla_value_shape ret_shape = dims_lhs;
  odla_value_shape reshape_dims = dims_rhs;
  if (dims_lhs.size > dims_rhs.size) {
    reshape_dims = BroadcastDims(dims_rhs, dims_lhs.size);
  } else if (dims_lhs.size < dims_rhs.size) {
    ret_shape = dims_rhs;
    reshape_dims = BroadcastDims(dims_lhs, dims_rhs.size);
  }
  for (int i = 0; i < ret_shape.size; i++) {
    if ((ret_shape.dims[i] > 0) && (reshape_dims.dims[i] > 0) &&
        (ret_shape.dims[i] < reshape_dims.dims[i])) {
      ret_shape.dims[i] = reshape_dims.dims[i];
    }
  }
  return ret_shape;
}

static odla_value binary_op(magicmind::IElementwise op_type, odla_value lhs,
                            odla_value rhs, const odla_value_id id) {
  magicmind::ITensor* lhs_tensor = lhs->tensor;
  magicmind::ITensor* rhs_tensor = rhs->tensor;
  const auto& dims_lhs = lhs->type.shape;
  const auto& dims_rhs = rhs->type.shape;
  auto out_dim = GetBroadcastOutputShape(dims_lhs, dims_rhs);
  auto op =
      g_comp->network->AddIElementwiseNode(lhs_tensor, rhs_tensor, op_type);

  return CreateValue(op, {lhs->type.element_type, out_dim}, id);
}

odla_value odla_Add(odla_value lhs, odla_value rhs, const odla_value_id id) {
  return binary_op(magicmind::IElementwise::ADD, lhs, rhs, id);
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
  magicmind::Layout mm_layout = GetMMLayout(input_layout);

  assert(mean->const_layer);
  assert(var->const_layer);

  auto mean_data = mean->const_layer->GetOutput(0);
  auto var_data = var->const_layer->GetOutput(0);
  magicmind::ITensor* scale_data =
      (scale && scale->const_layer ? scale->const_layer->GetOutput(0)
                                   : nullptr);
  magicmind::ITensor* offset_data =
      (offset && offset->const_layer ? offset->const_layer->GetOutput(0)
                                     : nullptr);

  auto bn = g_comp->network->AddIFusedBatchNormNode(
      input->tensor, mean_data, var_data, scale_data, offset_data);
  bn->SetEpsilon(epsilon);
  bn->SetAxis(1);
  //  bn->SetLayout(mm_layout, mm_layout);
  return CreateValue(bn, input->type, value_id);
}

odla_value odla_Conv(odla_value input, odla_memory_layout input_layout,
                     odla_uint32 group, odla_value kernel,
                     odla_memory_layout kernel_layout,
                     const odla_uint32* strides, const odla_uint32* dilations,
                     const odla_uint32* paddings_front,
                     const odla_uint32* paddings_back, odla_value bias,
                     odla_value_shape output_dims, const odla_value_id id) {
  auto weights_tensor = kernel->tensor; // const_layer->GetOutput(0);
  auto bias_tensor = (bias && bias->tensor ? bias->tensor : nullptr);
  auto conv =
      g_comp->network->AddIConvNode(input->tensor, weights_tensor, bias_tensor);
  CHECK(conv->SetStride(strides[0], strides[1]));
  CHECK(conv->SetPad(paddings_front[0], paddings_back[0], paddings_front[1],
                     paddings_back[1]));
  CHECK(conv->SetDilation(dilations[0], dilations[1]));
  CHECK(conv->SetPaddingMode(magicmind::IPaddingMode::EXPLICIT));
  auto mm_layout = GetMMLayout(input_layout);
  auto mm_weight_layout = GetMMLayout(kernel_layout);
  CHECK(conv->SetLayout(mm_layout, mm_weight_layout, mm_layout));

  auto conv_output = conv->GetOutput(0);
  // conv output tensor datatype should be set same with bias tensor
  if (bias_tensor) {
    CHECK(conv_output->SetDataType(bias_tensor->GetDataType()));
  } else {
    CHECK(conv_output->SetDataType(input->tensor->GetDataType()));
  }

  if (group > 1) {
    conv->SetGroup(static_cast<int64_t>(group));
  }
  auto type = input->type.element_type;
  if (bias && bias->tensor) type = bias->type.element_type;
  odla_value_type output_type{.element_type = input->type.element_type,
                              .shape = output_dims};
  auto ret = CreateValue(conv, output_type, id);

  return ret;
}

odla_value odla_Gemm(odla_value lhs, odla_bool transpose_lhs, odla_value rhs,
                     odla_bool transpose_rhs, odla_float32 alpha,
                     odla_float32 beta, odla_value bias,
                     odla_value_shape output_dims, const odla_value_id id) {
  auto bias_tensor = (bias && bias->tensor ? bias->tensor : nullptr);
  auto rhs_tensor = rhs->tensor;
  auto matmul =
      g_comp->network->AddIMatMulNode(lhs->tensor, rhs_tensor, bias_tensor);

  if (transpose_lhs) {
    matmul->SetTransA(true);
  } else {
    matmul->SetTransA(false);
  }
  if (transpose_rhs) {
    matmul->SetTransB(true);
  } else {
    matmul->SetTransB(false);
  }
  matmul->SetScalarAB(alpha);
  matmul->SetScalarC(beta);

  return CreateValue(matmul,
                     odla_value_type{lhs->type.element_type, output_dims}, id);
}

odla_value odla_MaxPool(odla_value input, odla_memory_layout input_layout,
                        const odla_uint32* window_dims,
                        const odla_uint32* strides,
                        const odla_uint32* paddings_front,
                        const odla_uint32* paddings_back,
                        odla_value_shape output_dims,
                        const odla_value_id value_id) {
  assert(input_layout == odla_memory_layout::ODLA_CHANNELS_FIRST);
  auto pooling = g_comp->network->AddIMaxPool2DNode(input->tensor, false);
  auto mm_layout = GetMMLayout(input_layout);
  pooling->SetPaddingMode(magicmind::IPaddingMode::EXPLICIT);
  pooling->SetKernel(window_dims[0], window_dims[1]);
  pooling->SetStride(strides[0], strides[1]);
  pooling->SetPad(paddings_front[0], paddings_back[0], paddings_front[1],
                  paddings_back[1]);
  pooling->SetLayout(mm_layout, mm_layout);
  pooling->SetCeilMode(0);

  odla_value_type output_type{.element_type = input->type.element_type,
                              .shape = output_dims};

  return CreateValue(pooling, output_type, value_id);
}

odla_value add_reduce_node(magicmind::IReduce type, odla_value input,
                           odla_size_t num_of_axes, const odla_uint32* axes,
                           odla_bool keep_dims, odla_value_shape output_dims,
                           const odla_value_id id) {
  std::vector<int> reduce_dims;
  for (int i = 0; i < num_of_axes; i++) {
    reduce_dims.push_back(axes[i]);
  }
  magicmind::IConstNode* reduce_shape = g_comp->network->AddIConstNode(
      magicmind::DataType::INT32, magicmind::Dims({(int64_t)num_of_axes}),
      &reduce_dims[0]);
  auto reduce = g_comp->network->AddIReduceNode(
      input->tensor, reduce_shape->GetOutput(0), type, false);

  if (keep_dims) {
    reduce->SetKeepDims(true);
  } else {
    reduce->SetKeepDims(false);
  }
  return CreateValue(reduce, {input->type.element_type, output_dims}, id);
}

odla_value odla_ReduceMean(odla_value input, odla_size_t num_of_axes,
                           const odla_uint32* axes, odla_bool keep_dims,
                           odla_value_shape output_dims,
                           const odla_value_id id) {
  return add_reduce_node(magicmind::IReduce::MEAN, input, num_of_axes, axes,
                         keep_dims, output_dims, id);
}

odla_value odla_ReduceMax(odla_value input, odla_size_t num_of_axes,
                          const odla_uint32* axes, odla_bool keep_dims,
                          odla_value_shape output_dims,
                          const odla_value_id id) {
  return add_reduce_node(magicmind::IReduce::MAX, input, num_of_axes, axes,
                         keep_dims, output_dims, id);
}

odla_value odla_ReduceSum(odla_value input, odla_size_t num_of_axes,
                          const odla_uint32* axes, odla_bool keep_dims,
                          odla_value_shape output_dims,
                          const odla_value_id id) {
  return add_reduce_node(magicmind::IReduce::ADD, input, num_of_axes, axes,
                         keep_dims, output_dims, id);
}

odla_value odla_Relu(odla_value input, const odla_value_id id) {
  auto relu = g_comp->network->AddIActivationNode(input->tensor,
                                                  magicmind::IActivation::RELU);
  return CreateValue(relu, input->type, id);
}

odla_value odla_Reshape(odla_value input, odla_value_shape output_dims,
                        const odla_value_id id) {
  std::vector<int> dims_data;
  if (output_dims.size > 0) {
    for (int i = 0; i < output_dims.size; i++) {
      dims_data.push_back((int)(output_dims.dims[i]));
    }
    magicmind::IConstNode* shape = g_comp->network->AddIConstNode(
        magicmind::DataType::INT32, magicmind::Dims({output_dims.size}),
        &dims_data[0]);
    magicmind::ITensor* shape_data = shape ? shape->GetOutput(0) : nullptr;
    auto reshape = g_comp->network->AddIReshapeNode(input->tensor, shape_data);

    return CreateValue(reshape, {input->type.element_type, output_dims}, id);
  } else {
    odla_uint32 axes = 0;
    return odla_Squeeze(input, 1, &axes, output_dims, id);
  }
}

odla_value odla_Clamp(odla_value input, odla_float32 lo, odla_float32 hi,
                      const odla_value_id id) {
  magicmind::IConstNode* hi_node = g_comp->network->AddIConstNode(
      magicmind::DataType::FLOAT32, magicmind::Dims({1}), &hi);
  magicmind::IConstNode* lo_node = g_comp->network->AddIConstNode(
      magicmind::DataType::FLOAT32, magicmind::Dims({1}), &lo);
  auto clip = g_comp->network->AddIClipNode(
      input->tensor, lo_node->GetOutput(0), hi_node->GetOutput(0));
  return CreateValue(clip, input->type, id);
}

odla_value odla_Gather(odla_value input, const odla_value indices,
                       odla_int32 axis, odla_value_shape output_dims,
                       const odla_value_id id) {
  axis = axis < 0 ? input->type.shape.size - 1 : axis;
  magicmind::IConstNode* axis_node = g_comp->network->AddIConstNode(
      magicmind::DataType::INT32, magicmind::Dims({1}), &axis);
  auto gather = g_comp->network->AddIGatherNode(input->tensor, indices->tensor,
                                                axis_node->GetOutput(0));
  return CreateValue(gather, {input->type.element_type, output_dims}, id);
}

odla_value odla_Mul(odla_value lhs, odla_value rhs, const odla_value_id id) {
  return binary_op(magicmind::IElementwise::MUL, lhs, rhs, id);
}

odla_value odla_Pow(odla_value base, odla_value exponent,
                    const odla_value_id id) {
  magicmind::ITensor* base_tensor = base->tensor;
  magicmind::ITensor* exp_tensor = exponent->tensor;
  const auto& dims_base = base->type.shape;
  const auto& dims_exp = exponent->type.shape;
  auto out_dim = GetBroadcastOutputShape(dims_base, dims_exp);
  auto pow = g_comp->network->AddIPowNode(base_tensor, exp_tensor);
  return CreateValue(pow, {base->type.element_type, out_dim}, id);
}

odla_value odla_Reciprocal(odla_value input, const odla_value_id id) {
  auto reciprocal = g_comp->network->AddIReciprocalNode(input->tensor);
  return CreateValue(reciprocal, input->type, id);
}

odla_value odla_Slice(odla_value input, const odla_int32* start,
                      const odla_int32* end, const odla_int32* stride,
                      odla_value_shape output_dims, const odla_value_id id) {
  const auto& dims = input->type.shape;
  magicmind::IConstNode* begin_node = g_comp->network->AddIConstNode(
      magicmind::DataType::INT32, magicmind::Dims({dims.size}), (void*)start);
  magicmind::INode* end_node = nullptr;
  bool is_dynamic_shape = false;
  for (int i = 0; i < output_dims.size; i++) {
    if (output_dims.dims[i] < 0) {
      is_dynamic_shape = true;
      break;
    }
  }
  if (is_dynamic_shape) {
    auto shape_node = g_comp->network->AddIShapeNode(input->tensor, nullptr);
    if (dims.size > 1) {
      std::vector<int> temp_data;
      temp_data.push_back(0);
      for (int i = 1; i < dims.size; i++) {
        temp_data.push_back((int)(dims.dims[i]) - *((int*)(end + i)));
      }
      auto temp_node = g_comp->network->AddIConstNode(
          magicmind::DataType::INT32, magicmind::Dims({dims.size}),
          &temp_data[0]);
      end_node = g_comp->network->AddIElementwiseNode(
          shape_node->GetOutput(0), temp_node->GetOutput(0),
          magicmind::IElementwise::SUB);
    } else {
      end_node = shape_node;
    }
  } else {
    end_node = g_comp->network->AddIConstNode(
        magicmind::DataType::INT32, magicmind::Dims({dims.size}), (void*)end);
  }
  magicmind::IConstNode* stride_node = g_comp->network->AddIConstNode(
      magicmind::DataType::INT32, magicmind::Dims({dims.size}), (void*)stride);
  auto slice = g_comp->network->AddIStridedSliceNode(
      input->tensor, begin_node->GetOutput(0), end_node->GetOutput(0),
      stride_node->GetOutput(0), nullptr);
  return CreateValue(slice, {input->type.element_type, output_dims}, id);
}

odla_value odla_Softmax(odla_value input, odla_int32 axis,
                        const odla_value_id id) {
  const auto& dims = input->type.shape;
  axis = axis < 0 ? dims.size - 1 : axis;
  magicmind::IConstNode* axis_node = g_comp->network->AddIConstNode(
      magicmind::DataType::INT32, magicmind::Dims({1}), &axis);
  auto sm =
      g_comp->network->AddISoftmaxNode(input->tensor, axis_node->GetOutput(0));
  return CreateValue(sm, input->type, id);
}

odla_value odla_Sqrt(odla_value input, const odla_value_id id) {
  auto sqrt = g_comp->network->AddISqrtNode(input->tensor);
  return CreateValue(sqrt, input->type, id);
}

odla_value odla_Sub(odla_value lhs, odla_value rhs, const odla_value_id id) {
  return binary_op(magicmind::IElementwise::SUB, lhs, rhs, id);
}

odla_value odla_Tanh(odla_value input, const odla_value_id id) {
  auto op = g_comp->network->AddIActivationNode(input->tensor,
                                                magicmind::IActivation::TANH);
  return CreateValue(op, input->type, id);
}

odla_value odla_Transpose(odla_value input, odla_value_shape permutations,
                          odla_value_shape output_dims,
                          const odla_value_id id) {
  auto permute_perms = g_comp->network->AddIConstNode(
      magicmind::DataType::INT64,
      magicmind::Dims({static_cast<int64_t>(permutations.size)}),
      permutations.dims);

  auto permute = g_comp->network->AddIPermuteNode(input->tensor,
                                                  permute_perms->GetOutput(0));

  return CreateValue(permute, {input->type.element_type, output_dims}, id);
}

odla_value odla_ArgMax(odla_value input, odla_int32 axis, odla_bool keep_dims,
                       odla_bool return_last_index,
                       odla_value_type output_value_type,
                       const odla_value_id id) {
  int reduce_axes = axis < 0 ? input->type.shape.size + axis : axis;
  auto dim_node = g_comp->network->AddIConstNode(
      magicmind::DataType::INT32, magicmind::Dims({1}), &reduce_axes);
  auto reduce_max = g_comp->network->AddIReduceNode(
      input->tensor, dim_node->GetOutput(0), magicmind::IReduce::MAX, true);
  return CreateValue(reduce_max->GetOutput(1), output_value_type, id);
}

odla_value odla_Cast(odla_value input, odla_element_type target_type,
                     const odla_value_id id) {
  auto mm_type = GetMMDataType(target_type);
  auto op = g_comp->network->AddICastNode(input->tensor, mm_type);
  odla_value_type dst_type{target_type, input->type.shape};
  return CreateValue(op, dst_type, id);
}

odla_value odla_Concat(odla_values inputs, odla_int32 axis,
                       odla_value_shape output_dims, const odla_value_id id) {
  int num = inputs.size;
  std::vector<magicmind::ITensor*> input_tensors(num);
  for (int i = 0; i < num; ++i) {
    input_tensors[i] = inputs.values[i]->tensor;
  }
  auto dim_node = g_comp->network->AddIConstNode(magicmind::DataType::INT32,
                                                 magicmind::Dims({1}), &axis);
  auto concat =
      g_comp->network->AddIConcatNode(dim_node->GetOutput(0), input_tensors);
  odla_value_type output_type{
      .element_type = inputs.values[0]->type.element_type,
      .shape = output_dims};
  return CreateValue(concat, output_type, id);
}

odla_value odla_CumSum(odla_value input, odla_value axis, odla_bool exclusion,
                       odla_bool reverse, const odla_value_id id) {
  assert(0 && "Unsupported cumsum method");
  return nullptr;
}

odla_value add_logic_node(magicmind::ILogic type, odla_value lhs,
                          odla_value rhs, const odla_value_id id) {
  auto op = g_comp->network->AddILogicNode(lhs->tensor, rhs->tensor, type);
  odla_value_shape out_dim;
  const auto& dims_lhs = lhs->type.shape;
  const auto& dims_rhs = rhs->type.shape;
  if (dims_lhs.size < dims_rhs.size) {
    out_dim = dims_rhs;
  } else {
    out_dim = dims_lhs;
  }
  return CreateValue(op, {lhs->type.element_type, out_dim}, id);
}

odla_value odla_Equal(odla_value lhs, odla_value rhs, const odla_value_id id) {
  return add_logic_node(magicmind::ILogic::EQ, lhs, rhs, id);
}

odla_value odla_GreaterOrEqual(odla_value lhs, odla_value rhs,
                               const odla_value_id id) {
  return add_logic_node(magicmind::ILogic::GE, lhs, rhs, id);
}

odla_value odla_Less(odla_value lhs, odla_value rhs, const odla_value_id id) {
  return add_logic_node(magicmind::ILogic::LT, lhs, rhs, id);
}

odla_value odla_LessOrEqual(odla_value lhs, odla_value rhs,
                            const odla_value_id id) {
  return add_logic_node(magicmind::ILogic::LE, lhs, rhs, id);
}

odla_value odla_And(odla_value lhs, odla_value rhs, const odla_value_id id) {
  return add_logic_node(magicmind::ILogic::AND, lhs, rhs, id);
}

odla_value odla_Or(odla_value lhs, odla_value rhs, const odla_value_id id) {
  return add_logic_node(magicmind::ILogic::OR, lhs, rhs, id);
}

odla_value odla_Not(odla_value input, const odla_value_id id) {
  auto op =
      g_comp->network->AddILogicNode(input->tensor, magicmind::ILogic::NOT);
  return CreateValue(op, input->type, id);
}

odla_value odla_Floor(odla_value input, const odla_value_id id) {
  auto floor = g_comp->network->AddIFloorNode(input->tensor);
  return CreateValue(floor, input->type, id);
}

odla_value odla_LogSoftmax(odla_value input, odla_int32 axis,
                           const odla_value_id id) {
  auto dim_node = g_comp->network->AddIConstNode(magicmind::DataType::INT32,
                                                 magicmind::Dims({1}), &axis);
  auto logsoftmax = g_comp->network->AddILogSoftmaxNode(input->tensor,
                                                        dim_node->GetOutput(0));
  return CreateValue(logsoftmax, input->type, id);
}

odla_value odla_Max(odla_value lhs, odla_value rhs, const odla_value_id id) {
  magicmind::ITensor* lhs_tensor = lhs->tensor;
  magicmind::ITensor* rhs_tensor = rhs->tensor;
  const auto& dims_lhs = lhs->type.shape;
  const auto& dims_rhs = rhs->type.shape;
  if (dims_lhs.size < dims_rhs.size) {
    auto temp = lhs_tensor;
    lhs_tensor = rhs_tensor;
    rhs_tensor = temp;
  }
  auto out_dim = GetBroadcastOutputShape(dims_lhs, dims_rhs);
  auto sub = g_comp->network->AddIMaximumNode(lhs_tensor, rhs_tensor);

  return CreateValue(sub, {lhs->type.element_type, out_dim}, id);
}

odla_value odla_Min(odla_value lhs, odla_value rhs, const odla_value_id id) {
  magicmind::ITensor* lhs_tensor = lhs->tensor;
  magicmind::ITensor* rhs_tensor = rhs->tensor;
  const auto& dims_lhs = lhs->type.shape;
  const auto& dims_rhs = rhs->type.shape;
  if (dims_lhs.size < dims_rhs.size) {
    auto temp = lhs_tensor;
    lhs_tensor = rhs_tensor;
    rhs_tensor = temp;
  }
  auto out_dim = GetBroadcastOutputShape(dims_lhs, dims_rhs);
  auto sub = g_comp->network->AddIMinimumNode(lhs_tensor, rhs_tensor);

  return CreateValue(sub, {lhs->type.element_type, out_dim}, id);
}

odla_value odla_OneHot(odla_value indices, odla_int32 depth, odla_value values,
                       odla_int32 axis, odla_value_shape output_dims,
                       const odla_value_id id) {
  auto data_type = GetMMDataType(values->type.element_type);
  int off_value = 0;
  auto off_value_node = g_comp->network->AddIConstNode(
      data_type, magicmind::Dims({1}), &off_value);
  auto depth_node = g_comp->network->AddIConstNode(
      magicmind::DataType::INT32, magicmind::Dims({1}), &depth);
  auto op = g_comp->network->AddIOneHotNode(
      indices->tensor, depth_node->GetOutput(0), values->tensor,
      off_value_node->GetOutput(0));
  op->SetAxis((int64_t)axis);
  return CreateValue(op, {values->type.element_type, output_dims}, id);
}

odla_value odla_Pad(odla_value input, const odla_uint32* padding_front,
                    const odla_uint32* padding_back,
                    odla_value_shape output_dims, const odla_value_id id) {
  const auto& input_dims = input->type.shape;
  assert(input_dims.dims[0] == output_dims.dims[0]);
  std::vector<int32_t> padding_data;
  for (int i = 0; i < input_dims.size; i++) {
    padding_data.push_back(padding_front[i]);
    padding_data.push_back(padding_back[i]);
  }
  auto pad_data_node = g_comp->network->AddIConstNode(
      magicmind::DataType::INT32, magicmind::Dims({input_dims.size, 2}),
      &padding_data[0]);
  auto op = g_comp->network->AddIPadNode(input->tensor,
                                         pad_data_node->GetOutput(0), nullptr);
  return CreateValue(op, {input->type.element_type, output_dims}, id);
}

odla_value odla_Div(odla_value lhs, odla_value rhs, const odla_value_id id) {
  auto op = g_comp->network->AddIDivNode(lhs->tensor, rhs->tensor);
  odla_value_shape out_dim;
  const auto& dims_lhs = lhs->type.shape;
  const auto& dims_rhs = rhs->type.shape;
  if (dims_lhs.size < dims_rhs.size) {
    out_dim = dims_rhs;
  } else {
    out_dim = dims_lhs;
  }

  return CreateValue(op, {lhs->type.element_type, out_dim}, id);
}

odla_value odla_Select(odla_value condition, odla_value a, odla_value b,
                       odla_value_shape output_dims, const odla_value_id id) {
  auto op =
      g_comp->network->AddISelectNode(a->tensor, b->tensor, condition->tensor);
  return CreateValue(op, {a->type.element_type, output_dims}, id);
}

odla_value odla_Shape(odla_value input, odla_value_shape output_dims,
                      const odla_value_id id) {
  auto op = g_comp->network->AddIShapeNode(input->tensor, nullptr);
  // todo data type need verify
  LOG_PRINT("todo: output data type need verify\n");
  return CreateValue(op, {input->type.element_type, output_dims}, id);
}

odla_value odla_Squeeze(odla_value input, odla_size_t num_of_axes,
                        const odla_uint32* axes, odla_value_shape output_dims,
                        const odla_value_id id) {
  std::vector<int32_t> axes_to_squeeze;
  for (int i = 0; i < num_of_axes; ++i) {
    axes_to_squeeze.emplace_back(axes[i]);
  }
  auto axes_data_node = g_comp->network->AddIConstNode(
      magicmind::DataType::INT32, magicmind::Dims({(int64_t)num_of_axes}),
      &axes_to_squeeze[0]);
  auto op = g_comp->network->AddISqueezeNode(input->tensor,
                                             axes_data_node->GetOutput(0));
  return CreateValue(op, {input->type.element_type, output_dims}, id);
}

odla_value odla_Tile(odla_value input, const odla_uint32* repeat,
                     odla_value_shape output_dims,
                     const odla_value_id value_id) {
  auto dims = input->type.shape.size;
  std::vector<int> repeat_data;
  for (int i = 0; i != dims; ++i) {
    repeat_data.push_back(repeat[i]);
  }
  auto repeat_data_node = g_comp->network->AddIConstNode(
      magicmind::DataType::INT32, magicmind::Dims({(int64_t)dims}),
      &repeat_data[0]);
  auto op = g_comp->network->AddITileNode(input->tensor,
                                          repeat_data_node->GetOutput(0));
  return CreateValue(op, {input->type.element_type, output_dims}, value_id);
}

odla_values odla_TopK(odla_value input, odla_uint32 K, odla_bool largest,
                      odla_bool sorted, odla_uint32 axis,
                      odla_value_type output_value_type,
                      odla_value_type output_value_index_type,
                      const odla_value_ids value_ids) {
  auto k_node = g_comp->network->AddIConstNode(magicmind::DataType::INT32,
                                               magicmind::Dims({1}), &K);
  auto topk =
      g_comp->network->AddITopKNode(input->tensor, k_node->GetOutput(0));
  topk->SetAxis((int64_t)axis);
  if (largest) {
    topk->SetLargest(true);
  } else {
    topk->SetLargest(false);
  }
  if (sorted) {
    topk->SetSorted(true);
  } else {
    topk->SetSorted(false);
  }
  return {.size = 2,
          .values = {CreateValue(topk->GetOutput(0), output_value_type,
                                 value_ids.value_ids[0]),
                     CreateValue(topk->GetOutput(1), output_value_index_type,
                                 value_ids.value_ids[1])}};
}

odla_value odla_LeakyRelu(odla_value input, odla_float32 alpha,
                          const odla_value_id id) {
  auto relu = g_comp->network->AddILeakyReluNode(input->tensor);
  relu->SetNegativeSlope(alpha);
  return CreateValue(relu, input->type, id);
}

odla_value odla_Sigmoid(odla_value input, const odla_value_id value_id) {
  auto relu = g_comp->network->AddIActivationNode(
      input->tensor, magicmind::IActivation::SIGMOID);
  return CreateValue(relu, input->type, value_id);
}

odla_value odla_Rsqrt(odla_value input, const odla_value_id id) {
  auto op = g_comp->network->AddIRsqrtNode(input->tensor);
  return CreateValue(op, input->type, id);
}

odla_status odla_BeginIf(odla_value condition, odla_value_id value_id) {
  branch_info br_info;
  br_info.branch = g_comp->network->AddIIfNode(condition->tensor);
  br_info.then_body = br_info.branch->CreateThenBody();
  br_info.else_body = br_info.branch->CreateElseBody();
  g_comp->branchs.push(br_info);
  return ODLA_SUCCESS;
}

odla_status odla_EnterBranchBody(odla_bool true_branch) {
  g_comp->branchs.top().in_true_body = true_branch != 0;
  if (g_comp->branchs.top().in_true_body) {
    g_comp->network =
        static_cast<magicmind::INetwork*>(g_comp->branchs.top().then_body);
  } else {
    g_comp->network =
        static_cast<magicmind::INetwork*>(g_comp->branchs.top().else_body);
  }
}

odla_values odla_EndIf(odla_value_ids value_ids) {
  g_comp->network = g_comp->network_backup;
  auto br_info = g_comp->branchs.top();
  int n = br_info.true_outputs.size();
  assert(n == br_info.false_outputs.size());
  assert(n <= ODLA_MAX_OUTPUTS);
  odla_values ret;
  ret.size = n;
  auto then_body = br_info.then_body;
  auto else_body = br_info.else_body;
  auto br = br_info.branch;
  g_comp->branchs.pop();
  for (int i = 0; i < n; ++i) {
    then_body->AddCondOutput(br_info.true_outputs[i]->tensor);
    else_body->AddCondOutput(br_info.false_outputs[i]->tensor);
  }
  for (int i = 0; i < n; ++i) {
    ret.values[i] = CreateValue(br->GetOutput(i), br_info.true_outputs[i]->type,
                                value_ids.value_ids[i]);
  }
  return ret;
}

odla_value odla_SliceDynamic(odla_value input, odla_value start,
                             odla_value size, odla_value stride,
                             odla_value_shape output_dims,
                             const odla_value_id id) {
  auto slice = g_comp->network->AddISliceNode(input->tensor, start->tensor,
                                              size->tensor);
  return CreateValue(slice, {input->type.element_type, output_dims}, id);
}

odla_value odla_Stack(odla_values inputs, odla_int32 axis,
                      odla_value_shape output_shape, const odla_value_id id) {
  int num = inputs.size;
  std::vector<magicmind::ITensor*> input_tensors(num);
  for (int i = 0; i < num; ++i) {
    input_tensors[i] = inputs.values[i]->tensor;
  }

  auto dim_node = g_comp->network->AddIConstNode(magicmind::DataType::INT32,
                                                 magicmind::Dims({1}), &axis);
  auto pack =
      g_comp->network->AddIPackNode(dim_node->GetOutput(0), input_tensors);
  return CreateValue(pack, {inputs.values[0]->type.element_type, output_shape},
                     id);
}
odla_value odla_ReduceL2(odla_value input, odla_size_t num_of_axes,
                         const odla_uint32* axes, odla_bool keep_dims,
                         odla_float32 epsilon, odla_value_shape output_dims,
                         const odla_value_id value_id) {
  if (num_of_axes == 1) {
    auto dim_node = g_comp->network->AddIConstNode(
        magicmind::DataType::INT32, magicmind::Dims({1}), (int*)axes);
    auto normalize = g_comp->network->AddINormalizeNode(
        input->tensor, dim_node->GetOutput(0), nullptr);
    normalize->SetEpsilon(epsilon);
    return CreateValue(normalize, {input->type.element_type, output_dims},
                       value_id);
  } else {
    return input;
  }
}

int GetMMDetectionCodeType(int code_type) {
  switch (code_type) {
    case 1:
      return 0;
    case 2:
      return 1;
    case 3:
      return 2;
  }
}

odla_values odla_CustomOp(odla_values inputs, const odla_char* op_name,
                          const odla_char* function_name,
                          const odla_value_ids ids, ...) {
  if (std::string(op_name) == "DetectionOutput") {
    assert(ids.size == 1);
    assert(inputs.size == 3);
    const char* id = reinterpret_cast<const char*>(ids.value_ids[0]);
    const auto& name = id != nullptr ? std::string(id) : "DetectionOutput";
    va_list p_args;
    va_start(p_args, ids);
    int bg_label_id = va_arg(p_args, int);
    float confidence_threshold = (float)va_arg(p_args, double);
    int keep_topk = va_arg(p_args, int);
    int classes = va_arg(p_args, int);
    bool share_loc = (bool)va_arg(p_args, int);
    int code_type = va_arg(p_args, int);
    float nms_threshold = (float)va_arg(p_args, double);
    int nms_topk = va_arg(p_args, int);
    int nms_eta = va_arg(p_args, int);
    va_end(p_args);

    // reshape loc
    std::vector<int32_t> perms_vec = {0, 2, 1};
    auto permute_perms = g_comp->network->AddIConstNode(
        magicmind::DataType::INT32,
        magicmind::Dims({static_cast<int64_t>(perms_vec.size())}),
        perms_vec.data());
    std::vector<int32_t> loc_shape_vec = {-1, 4};
    if (!share_loc) {
      loc_shape_vec[1] = classes * 4;
    }
    auto loc_reshape_shape = g_comp->network->AddIConstNode(
        magicmind::DataType::INT32,
        magicmind::Dims({static_cast<int64_t>(loc_shape_vec.size())}),
        loc_shape_vec.data());
    auto loc_reshape_node = g_comp->network->AddIReshapeNode(
        inputs.values[0]->tensor, loc_reshape_shape->GetOutput(0));
    CHECK(loc_reshape_node->SetAxis(1));
    CHECK(loc_reshape_node->SetNumAxes(-1));
    CHECK(loc_reshape_node->SetAllowZero(false));
    // permute loc
    auto loc_permute = g_comp->network->AddIPermuteNode(
        loc_reshape_node->GetOutput(0), permute_perms->GetOutput(0));

    // reshape conf
    std::vector<int32_t> conf_shape_vec = {-1, classes};
    auto conf_reshape_shape = g_comp->network->AddIConstNode(
        magicmind::DataType::INT32,
        magicmind::Dims({static_cast<int64_t>(conf_shape_vec.size())}),
        conf_shape_vec.data());
    auto conf_reshape_node = g_comp->network->AddIReshapeNode(
        inputs.values[1]->tensor, conf_reshape_shape->GetOutput(0));
    CHECK(conf_reshape_node->SetAxis(1));
    CHECK(conf_reshape_node->SetNumAxes(-1));
    CHECK(conf_reshape_node->SetAllowZero(false));
    // permute conf
    auto conf_permute = g_comp->network->AddIPermuteNode(
        conf_reshape_node->GetOutput(0), permute_perms->GetOutput(0));

    // reshape prior
    std::vector<int32_t> prior_shape_vec = {2, -1, 4};
    auto prior_reshape_shape = g_comp->network->AddIConstNode(
        magicmind::DataType::INT32,
        magicmind::Dims({static_cast<int64_t>(prior_shape_vec.size())}),
        prior_shape_vec.data());
    auto prior_reshape_node = g_comp->network->AddIReshapeNode(
        inputs.values[2]->tensor, prior_reshape_shape->GetOutput(0));
    CHECK(prior_reshape_node->SetAxis(0));
    CHECK(prior_reshape_node->SetNumAxes(-1));
    CHECK(prior_reshape_node->SetAllowZero(false));
    // permute prior
    auto prior_permute = g_comp->network->AddIPermuteNode(
        prior_reshape_node->GetOutput(0), permute_perms->GetOutput(0));

    std::vector<magicmind::ITensor*> det_input = {loc_permute->GetOutput(0),
                                                  conf_permute->GetOutput(0)};
    auto detection_output = g_comp->network->AddIDetectionOutputNode(
        det_input, prior_permute->GetOutput(0));
    detection_output->SetAlgo(magicmind::IDetectionOutputAlgo::SSD);
    detection_output->SetBackgroundLabelId((int64_t)bg_label_id);
    detection_output->SetCodeType(
        GetMMDetectionCodeType(code_type)); // CENTER_SIZE
    detection_output->SetConfidenceThresh(confidence_threshold);
    detection_output->SetKeepTopK(keep_topk);
    detection_output->SetNmsThresh(nms_threshold);
    detection_output->SetNmsTopK(nms_topk);
    detection_output->SetNumClass((int64_t)classes);
    odla_values ret;
    ret.size = 2;
    odla_value_shape shape1{.size = 1, .dims = {1}};
    ret.values[0] = CreateValue(detection_output->GetOutput(1),
                                {inputs.values[0]->type.element_type, shape1},
                                (const odla_value_id) "detection-output-size");
    ret.values[1] = CreateValue(detection_output->GetOutput(0),
                                inputs.values[0]->type, ids.value_ids[0]);
    return ret;
  } else {
    printf("not support %s\n", op_name);
    assert(0);
  }
}

odla_values odla_Split(odla_value input, odla_value split_dim,
                       odla_int32 num_split, const odla_value_ids value_ids) {
  auto split_node = g_comp->network->AddISplitNode(
      input->tensor, split_dim->tensor, (int64_t(num_split)));
  odla_values ret;
  ret.size = num_split;
  for (int i = 0; i < ret.size; i++) {
    ret.values[i] = CreateValue(split_node->GetOutput(i), input->type,
                                value_ids.value_ids[i]);
  }
  return ret;
}
odla_value odla_LpNormalize(odla_value input, odla_int32 p,
                            odla_memory_layout input_layout,
                            odla_size_t axes_size, const odla_int32* axes,
                            odla_float32 epsilon, odla_value scale,
                            const odla_value_id value_id) {
  auto dim_node = g_comp->network->AddIConstNode(
      magicmind::DataType::INT32, magicmind::Dims({(int64_t)axes_size}),
      (int*)axes);
  auto scale_tensor = scale ? scale->tensor : nullptr;
  int scale_total_size = 1;
  if (scale_tensor) {
    if (scale->type.shape.size == 1) {
      scale_total_size = scale->type.shape.dims[0];
    } else if (scale->type.shape.size != 0) {
      for (int i = 0; i < scale->type.shape.size; i++) {
        scale_total_size *= scale->type.shape.dims[i];
      }
      auto shape_data_node = g_comp->network->AddIConstNode(
          magicmind::DataType::INT32, magicmind::Dims({1}), &scale_total_size);
      auto reshape_node = g_comp->network->AddIReshapeNode(
          scale_tensor, shape_data_node->GetOutput(0));
      scale_tensor = reshape_node->GetOutput(0);
    }
  }
  auto normalize = g_comp->network->AddINormalizeNode(
      input->tensor, dim_node->GetOutput(0), scale_tensor);
  normalize->SetP((float)p);
  normalize->SetEpsilon(epsilon);
  if (scale_tensor) {
    // caffe case
    if (axes_size == 1) {
      normalize->SetAcrossSpatial(
          magicmind::INormalizeAcrossSpatialMode::NOT_ACROSS_SPATIAL);
    } else {
      normalize->SetAcrossSpatial(
          magicmind::INormalizeAcrossSpatialMode::ACROSS_SPATIAL);
    }

    if (scale_total_size == 1) {
      normalize->SetChannelShared(
          magicmind::INormalizeChannelSharedMode::CHANNEL_SHARED);
    } else {
      normalize->SetChannelShared(
          magicmind::INormalizeChannelSharedMode::NOT_CHANNEL_SHARED);
    }
  } else {
    normalize->SetAcrossSpatial(
        magicmind::INormalizeAcrossSpatialMode::ACROSS_AXIS_TENSOR);
  }
  return CreateValue(normalize, input->type, value_id);
}

} // C extern
