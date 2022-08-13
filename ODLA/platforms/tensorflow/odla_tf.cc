//===- odla_tf.cc ---------------------------------------------------------===//
//
// Copyright (C) 2019-2022 Alibaba Group Holding Limited.
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

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <unordered_map>

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/public/session.h"

using namespace std;
using namespace tensorflow;

#define LOG_TF_ERROR(status)                             \
  do {                                                   \
    if (!status.ok()) {                                  \
      LOG(ERROR) << status.error_message() << std::endl; \
    }                                                    \
  } while (0);

#define TF_RET_CHECK(status) \
  do {                       \
    if (!status.ok()) {      \
      LOG_TF_ERROR(status);  \
      return ODLA_FAILURE;   \
    }                        \
  } while (0);

class UnownedBuffer : public tensorflow::TensorBuffer {
  std::size_t len_;

 public:
  UnownedBuffer(void* data, std::size_t len)
      : len_(len), tensorflow::TensorBuffer(data) {}
  ~UnownedBuffer() override {}
  // returns how many bytes we have in our buffer
  std::size_t size() const override { return len_; };
  // needed so TF knows this isn't a child of some other buffer
  TensorBuffer* root_buffer() override { return this; }
  void FillAllocationDescription(
      tensorflow::AllocationDescription* proto) const override{};
  bool OwnsMemory() const override { return false; }
};

struct _odla_device {};

static size_t GetElemSize(tensorflow::DataType dt) {
  switch (dt) {
    case tensorflow::DT_BOOL:
    case tensorflow::DT_UINT8:
    case tensorflow::DT_INT8:
    case tensorflow::DT_QINT8:
    case tensorflow::DT_QUINT8:
      return 1;
    case tensorflow::DT_BFLOAT16:
    case tensorflow::DT_QINT16:
    case tensorflow::DT_QUINT16:
    case tensorflow::DT_UINT16:
    case tensorflow::DT_HALF:
    case tensorflow::DT_INT16:
      return 2;
    case tensorflow::DT_QINT32:
    case tensorflow::DT_INT32:
    case tensorflow::DT_UINT32:

    case tensorflow::DT_FLOAT:
      return 4;
    case tensorflow::DT_INT64:
    case tensorflow::DT_UINT64:
    case tensorflow::DT_DOUBLE:
      return 8;
      // TF_STRING = 7,
      // TF_COMPLEX64 = 8,  // Single-precision complex
      // TF_COMPLEX = 8,    // Old identifier kept for API backwards
      // compatibility TF_COMPLEX128 = 18,  // Double-precision complex
      // TF_RESOURCE = 20, TF_VARIANT = 21
  }
  LOG(ERROR) << "Unhandled data type" << std::endl;
  return 0;
}

static int64_t GetTotalElements(const odla_value_shape& dims) {
  return dims.size == 0 ? 1
                        : std::accumulate(dims.dims, dims.dims + dims.size, 1,
                                          std::multiplies<size_t>());
}

static int GetODLADataType(tensorflow::DataType tf_dt) {
  static const std::unordered_map<tensorflow::DataType, odla_element_type>
      tf2odla({{tensorflow::DT_BOOL, ODLA_BOOL},
               {tensorflow::DT_UINT8, ODLA_UINT8},
               {tensorflow::DT_INT8, ODLA_INT8},
               {tensorflow::DT_QINT8, ODLA_QINT8},
               {tensorflow::DT_QUINT8, ODLA_QUINT8},
               {tensorflow::DT_BFLOAT16, ODLA_BFLOAT16},
               {tensorflow::DT_QINT16, ODLA_QINT16},
               {tensorflow::DT_QUINT16, ODLA_QUINT16},
               {tensorflow::DT_UINT16, ODLA_UINT16},
               {tensorflow::DT_HALF, ODLA_FLOAT16},
               {tensorflow::DT_INT16, ODLA_INT16},
               {tensorflow::DT_QINT32, ODLA_QINT32},
               {tensorflow::DT_INT32, ODLA_INT32},
               {tensorflow::DT_UINT32, ODLA_UINT32},
               {tensorflow::DT_FLOAT, ODLA_FLOAT32},
               {tensorflow::DT_INT64, ODLA_INT64},
               {tensorflow::DT_UINT64, ODLA_UINT64},
               {tensorflow::DT_DOUBLE, ODLA_FLOAT64},
               {tensorflow::DT_STRING, ODLA_STRING}});
  auto it = tf2odla.find(tf_dt);
  return it == tf2odla.end() ? -1 : it->second;
}

static size_t ComputeMemorySize(tensorflow::DataType dt,
                                const tensorflow::TensorShape& shape) {
  return shape.num_elements() * GetElemSize(dt);
}

static size_t ComputeMemorySize(const tensorflow::Tensor& tensor) {
  return tensor.TotalBytes();
}

struct _odla_value {
  _odla_value(odla_computation comp, tensorflow::DataType dt,
              const tensorflow::TensorShape& shape, const std::string& name)
      : comp(comp), dtype(dt), shape(shape), name(name) {
    mem_size =
        dt == tensorflow::DT_RESOURCE_REF ? 0 : Tensor(dt, shape).TotalBytes();
  }
  odla_computation comp;
  std::string name;
  tensorflow::TensorShape shape;
  tensorflow::DataType dtype;
  size_t mem_size;
};

struct _odla_computation {
  tensorflow::GraphDef graph_def;
  std::vector<std::unique_ptr<_odla_value>> inputs;
  std::vector<std::unique_ptr<_odla_value>> outputs;
};

struct _odla_context {
  _odla_context(tensorflow::Session* tf_session)
      : initialized(false), session(tf_session) {}

  ~_odla_context() {
    input_tensors.clear();
    session->Close();
  }

  bool initialized;
  tensorflow::Session* session;
  tensorflow::RunOptions run_options;
  std::vector<std::pair<std::string, tensorflow::Tensor>> input_tensors;
  std::vector<std::string> output_names;
  std::vector<tensorflow::Tensor> output_tensors;
};

static std::unordered_map<odla_context, std::unique_ptr<_odla_context>> g_ctxs;
static std::unordered_map<odla_computation, std::unique_ptr<_odla_computation>>
    g_comps;
static odla_computation g_comp;

static _odla_device g_device;

odla_status odla_AllocateDevice(odla_vendor vendor,
                                odla_device_name device_name,
                                odla_int32_t device_idx, odla_device* device) {
  *device = &g_device;
  return ODLA_SUCCESS;
}

odla_status odla_CreateComputation(odla_computation* comp) {
  return ODLA_FAILURE;
}

odla_status odla_LoadComputation(const odla_char* file_name,
                                 odla_computation* computation) {
  *computation = nullptr;
  auto comp = std::make_unique<_odla_computation>();

  auto s = ReadBinaryProto(tensorflow::Env::Default(), file_name,
                           &(comp->graph_def));
  TF_RET_CHECK(s);
  const std::string output_name_prefix = "ret_";
  for (const auto& node : comp->graph_def.node()) {
    if (node.op() == "Placeholder") {
      odla_value val;
      {
        const auto& shape = node.attr().at("shape").shape();
        auto dt = node.attr().at("dtype").type();
        auto v =
            std::make_unique<_odla_value>(comp.get(), dt, shape, node.name());
        val = v.get();
        comp->inputs.push_back(std::move(v));
      }
    } else if (node.op() == "Identity" ||
               node.name().substr(0, output_name_prefix.size()) ==
                   output_name_prefix) {
      auto v =
          std::make_unique<_odla_value>(comp.get(), tensorflow::DT_RESOURCE_REF,
                                        tensorflow::TensorShape(), node.name());
      comp->outputs.push_back(std::move(v));
    }
  }
  *computation = comp.get();
  g_comps[*computation] = std::move(comp);
  return ODLA_SUCCESS;
}

odla_status odla_GetNumOfArgsFromComputation(const odla_computation computation,
                                             odla_uint32* num_args) {
  *num_args = computation->inputs.size();
  return ODLA_SUCCESS;
}

odla_status odla_GetArgFromComputationByIdx(const odla_computation computation,
                                            const odla_uint32 arg_idx,
                                            odla_value* arg_value) {
  if (arg_idx >= computation->inputs.size()) {
    *arg_value = nullptr;
    return ODLA_FAILURE;
  }
  *arg_value = computation->inputs[arg_idx].get();
  return ODLA_SUCCESS;
}

odla_status odla_GetNumOfOutputsFromComputation(
    const odla_computation computation, odla_uint32* num_outputs) {
  *num_outputs = computation->outputs.size();
  return ODLA_SUCCESS;
}

odla_status odla_GetOutputFromComputationByIdx(
    const odla_computation computation, const odla_uint32 arg_idx,
    odla_value* arg_value) {
  if (arg_idx >= computation->outputs.size()) {
    *arg_value = nullptr;
    return ODLA_FAILURE;
  }
  *arg_value = computation->outputs[arg_idx].get();
  return ODLA_SUCCESS;
}

odla_status odla_GetValueId(const odla_value value, odla_value_id* value_id) {
  *value_id =
      reinterpret_cast<odla_value_id>(const_cast<char*>(value->name.c_str()));
  return ODLA_SUCCESS;
}

odla_status odla_GetValueType(const odla_value value,
                              odla_value_type* value_type) {
  auto dt = GetODLADataType(value->dtype);
  if (dt < 0 || value->shape.dims() > ODLA_MAX_DIMENSION) {
    return ODLA_FAILURE;
  }
  value_type->element_type = static_cast<odla_element_type>(dt);
  value_type->shape.size = value->shape.dims();
  for (int i = 0; i < value_type->shape.size; ++i) {
    value_type->shape.dims[i] = value->shape.dim_size(i);
  }
  return ODLA_SUCCESS;
}

static inline std::string GetTFOutputName(const std::string& name) {
  return name + ":0";
}

static bool InitContext(odla_context ctx, const _odla_computation& comp) {
  if (ctx->initialized) {
    return true;
  }
  ctx->input_tensors.reserve(comp.inputs.size());
  for (const auto& input : comp.inputs) {
    ctx->input_tensors.push_back(
        {input->name, tensorflow::Tensor(input->dtype, input->shape)});
  }
  ctx->output_names.reserve(comp.outputs.size());
  for (const auto& output : comp.outputs) {
    ctx->output_names.push_back(GetTFOutputName(output->name));
  }
  auto status = ctx->session->Create(comp.graph_def);
  LOG_TF_ERROR(status);
  ctx->initialized = status.ok();
  return ctx->initialized;
}

odla_status odla_BindToArgumentById(const odla_value_id value_id,
                                    const odla_void* data_ptr,
                                    odla_context context) {
  return ODLA_FAILURE;
}

odla_status odla_BindToArgument(odla_value value, const odla_void* data_ptr,
                                odla_context context) {
  if (!InitContext(context, *value->comp)) {
    return ODLA_FAILURE;
  }
  auto it =
      std::find_if(context->input_tensors.begin(), context->input_tensors.end(),
                   [&value](const auto& v) { return v.first == value->name; });
  if (it == context->input_tensors.end()) {
    return ODLA_FAILURE;
  }
  UnownedBuffer* buf =
      new UnownedBuffer(const_cast<void*>(data_ptr), value->mem_size);
  // buf has it's own ref counter and is owned by Tensor.
  it->second = Tensor(value->dtype, value->shape, buf);
  return ODLA_SUCCESS;
}

odla_status odla_CreateContext(odla_context* context) {
  *context = nullptr;
  tensorflow::Session* tf_session = nullptr;
  tensorflow::SessionOptions session_options;
  tensorflow::Status status =
      tensorflow::NewSession(session_options, &tf_session);
  TF_RET_CHECK(status);
  auto ctx = std::make_unique<_odla_context>(tf_session);

  *context = ctx.get();
  g_ctxs[*context] = std::move(ctx);
  return ODLA_SUCCESS;
}

odla_status odla_ExecuteComputation(const odla_computation computation,
                                    const odla_context context,
                                    const odla_compute_mode mode,
                                    odla_device device) {
  if (!InitContext(context, *computation)) {
    return ODLA_FAILURE;
  }
  auto status = context->session->Run(
      context->run_options, context->input_tensors, context->output_names,
      /* target node */ {}, &context->output_tensors, nullptr /*run_metadata*/);
  TF_RET_CHECK(status);
  return ODLA_SUCCESS;
}

static const tensorflow::Tensor* GetOutputTensor(odla_context ctx,
                                                 odla_value value) {
  auto it = std::find_if(ctx->output_names.begin(), ctx->output_names.end(),
                         [&value](const auto& name) {
                           return name == GetTFOutputName(value->name);
                         });
  if (it == ctx->output_names.end()) return nullptr;
  auto idx = it - ctx->output_names.begin();
  if (idx >= ctx->output_tensors.size()) {
    return nullptr;
  }
  return &ctx->output_tensors[idx];
}

odla_status odla_GetRuntimeShape(odla_context context, odla_value value,
                                 odla_value_shape* value_shape_ptr) {
  const tensorflow::Tensor* tensor = GetOutputTensor(context, value);
  if (tensor == nullptr || !tensor->IsInitialized()) return ODLA_FAILURE;

  value_shape_ptr->size = tensor->shape().dims();
  for (int i = 0; i < tensor->shape().dims(); ++i) {
    value_shape_ptr->dims[i] = tensor->shape().dim_size(i);
  }
  return ODLA_SUCCESS;
}

odla_status odla_BindToOutputById(const odla_value_id value_id,
                                  odla_void* data_ptr, odla_context context) {
  return ODLA_FAILURE;
}

odla_status odla_BindToOutput(odla_value value, odla_void* data_ptr,
                              odla_context context) {
  const tensorflow::Tensor* tensor = GetOutputTensor(context, value);
  if (tensor == nullptr || !tensor->IsInitialized()) {
    return ODLA_FAILURE;
  }
  const void* raw_data = nullptr;
  switch (tensor->dtype()) {
    case DT_FLOAT:
      raw_data = tensor->flat<float>().data();
      break;
    case DT_INT32:
      raw_data = tensor->flat<int32_t>().data();
      break;
    case DT_HALF:
      raw_data = tensor->flat<short>().data();
      break;
    default:
      LOG(ERROR) << "Unhandled data type in odla_BindToOutput"
                 << tensor->dtype();
      break;
  }
  if (raw_data == nullptr) {
    return ODLA_FAILURE;
  }
  size_t len = ComputeMemorySize(*tensor);
  memcpy(data_ptr, raw_data, len);
  return ODLA_SUCCESS;
}

odla_status odla_DestroyContext(odla_context context) {
  auto it = g_ctxs.find(context);
  if (it == g_ctxs.end()) {
    return ODLA_FAILURE;
  }
  g_ctxs.erase(it);
  return ODLA_SUCCESS;
}

odla_status odla_DestroyComputation(odla_computation computation) {
  auto it = g_comps.find(computation);
  if (it == g_comps.end()) {
    return ODLA_FAILURE;
  }
  g_comps.erase(it);
  return ODLA_SUCCESS;
}

odla_status odla_DestroyDevice(odla_device device) { return ODLA_SUCCESS; }

template <typename T>
static void print_result(tensorflow::Tensor tensor, ofstream& os) {
  auto flat_tensor = tensor.flat<T>();
  for (int i = 0; i < flat_tensor.size(); ++i) {
    os << flat_tensor(i) << std::endl;
  }
}

#define ODLA_RET_CHECK(status)                                           \
  do {                                                                   \
    if (status != ODLA_SUCCESS) {                                        \
      LOG(ERROR) << "ODLA API failed in " << __func__ << "(" << __FILE__ \
                 << ":" << __LINE__ << std::endl;                        \
      exit(1);                                                           \
    }                                                                    \
  } while (0);

// Test code
#ifdef TEST_ODLA_TF
int main(int argc, char* argv[]) {
  odla_device dev = nullptr;
  odla_computation comp = nullptr;
  odla_context ctx = nullptr;
  odla_status status;
  ODLA_RET_CHECK(odla_AllocateDevice(0, ODLA_DEVICE_DEFAULT, &dev, nullptr));
  ODLA_RET_CHECK(odla_LoadComputation(argv[1], &comp));
  odla_uint32 num_inputs = 0;
  ODLA_RET_CHECK(odla_GetNumOfArgsFromComputation(comp, &num_inputs));
  std::cout << "Num of inputs:" << num_inputs << std::endl;
  std::vector<std::unique_ptr<float[]>> float_inputs(num_inputs);

  ODLA_RET_CHECK(odla_CreateContext(&ctx));
  constexpr int alignment = 64;
  for (int idx = 0; idx < num_inputs; ++idx) {
    odla_value v = nullptr;
    ODLA_RET_CHECK(odla_GetArgFromComputationByIdx(comp, idx, &v));
    odla_value_id id;
    odla_value_type ty;
    ODLA_RET_CHECK(odla_GetValueId(v, &id));
    ODLA_RET_CHECK(odla_GetValueType(v, &ty));
    int64_t num_elems = 1;
    std::cout << "  " << reinterpret_cast<const char*>(id) << " "
              << ty.element_type << "[";

    for (int i = 0, e = ty.shape.size; i < e; ++i) {
      num_elems *= ty.shape.dims[i];
      std::cout << ty.shape.dims[i] << " ";
    }
    std::cout << "]" << std::endl;
    float_inputs[idx].reset(new (std::align_val_t(alignment)) float[num_elems]);
    for (int i = 0; i < num_elems; ++i) float_inputs[idx][i] = 1.0F;
    ODLA_RET_CHECK(odla_BindToArgument(v, float_inputs[idx].get(), ctx));
  }
  odla_uint32 num_outputs = 0;
  ODLA_RET_CHECK(odla_GetNumOfOutputsFromComputation(comp, &num_outputs));
  std::cout << "Num of outputs:" << num_outputs << std::endl;

  for (int idx = 0; idx < num_outputs; ++idx) {
    odla_value v = nullptr;
    ODLA_RET_CHECK(odla_GetOutputFromComputationByIdx(comp, idx, &v));
    odla_value_id id;
    ODLA_RET_CHECK(odla_GetValueId(v, &id));
    std::cout << "  " << reinterpret_cast<const char*>(id) << std::endl;
  }

  ODLA_RET_CHECK(
      odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, dev));
  odla_value_shape s;
  std::vector<float> out;
  for (int idx = 0; idx < num_outputs; ++idx) {
    odla_value v = nullptr;
    ODLA_RET_CHECK(odla_GetOutputFromComputationByIdx(comp, idx, &v));
    ODLA_RET_CHECK(odla_GetRuntimeShape(ctx, v, &s));
    std::cout << s.size << ": " << std::endl;
    for (int i = 0; i < s.size; ++i) std::cout << s.dims[i] << " ";
    std::cout << endl;
    out.resize(GetTotalElements(s));
    ODLA_RET_CHECK(odla_BindToOutput(v, out.data(), ctx));
    for (auto x : out) {
      std::cout << x << " ";
    }
  }
  ODLA_RET_CHECK(
      odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, dev));

  odla_DestroyContext(ctx);
  odla_DestroyComputation(comp);
  odla_DestroyDevice(dev);
  return 0;
}
#endif