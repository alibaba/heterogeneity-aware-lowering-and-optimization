//===- odla_torchscript.cc ------------------------------------------------===//
//
// Copyright (C) 2022 Alibaba Group Holding Limited.
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
#include <c10/util/ArrayRef.h>
#include <torch/csrc/jit/serialization/import.h>

#include <ODLA/odla.h>

#include <sstream>

const uint32_t MAX_OUTPUT_TENSORS = 10;
const uint32_t MAX_INPUT_TENSORS = 20;

struct _odla_device {
  c10::DeviceType device_t_;
};

struct _odla_value {
  _odla_value(uint32_t v) : id_(v) {}
  uint32_t id_;
};

struct _odla_executable {
  torch::jit::Module module_;
  std::vector<odla_value> odla_inputs_outputs_;
  odla_uint32 num_inputs_;
};

struct _odla_context {
  _odla_context();
  std::vector<torch::jit::IValue> inputs_;
  std::vector<odla_value_type> input_types_;
  torch::jit::IValue output_;
  std::vector<at::Tensor> output_tensors_;
  odla_uint32 num_output_tensors_;
};

_odla_context::_odla_context() {
  inputs_.resize(MAX_INPUT_TENSORS);
  input_types_.resize(MAX_INPUT_TENSORS);
}

size_t static getElementCount(const odla_value_shape& dims) {
  return dims.size == 0 ? 1
                        : std::accumulate(dims.dims, dims.dims + dims.size, 1,
                                          std::multiplies<size_t>());
}

c10::IntArrayRef static toTensorDim(odla_value_shape& dims) {
  return dims.size == 0 ? c10::IntArrayRef(1)
                        : c10::IntArrayRef(dims.dims, dims.size);
}

c10::ScalarType static toTensorDataType(odla_element_type dt) {
  static const std::unordered_map<odla_element_type, c10::ScalarType> dt_map = {
      {ODLA_FLOAT32, c10::ScalarType::Float},
      {ODLA_INT32, c10::ScalarType::Int},
      {ODLA_BOOL, c10::ScalarType::Bool}};
  auto it = dt_map.find(dt);
  return it == dt_map.end() ? c10::ScalarType::Float : it->second;
}

odla_element_type static toODLADataType(const c10::ScalarType& st) {
  static const std::unordered_map<c10::ScalarType, odla_element_type> dt_map = {
      {c10::ScalarType::Float, ODLA_FLOAT32},
      {c10::ScalarType::Int, ODLA_INT32},
      {c10::ScalarType::Bool, ODLA_BOOL}};
  auto it = dt_map.find(st);
  return it == dt_map.end() ? ODLA_FLOAT32 : it->second;
}

odla_value_type static toODLAValueType(const c10::ScalarType& dt,
                                       at::IntArrayRef dims) {
  odla_value_type ty;
  ty.element_type = toODLADataType(dt);
  ty.shape.size = dims.size();
  int i = 0;
  for (auto d : dims) {
    ty.shape.dims[i++] = d;
  }
  return ty;
}

static std::unordered_map<odla_context, std::unique_ptr<_odla_context>> g_ctxs;
static std::unordered_map<odla_executable, std::unique_ptr<_odla_executable>>
    g_executables;

static _odla_device g_device{c10::kCUDA};

odla_status odla_AllocateDevice(const odla_vendor vendor,
                                const odla_device_name device_name,
                                odla_device* device, const char* config) {
  *device = &g_device;
  return ODLA_SUCCESS;
}

odla_status odla_LoadExecutable(odla_resource_location location,
                                odla_device device,
                                odla_executable* computation) {
  *computation = nullptr;
  if (location.location_type != ODLA_LOCATION_MEMORY &&
      location.location_type != ODLA_LOCATION_PATH) {
    return ODLA_FAILURE;
  }
  auto comp = std::make_unique<_odla_executable>();
  if (location.location_type == ODLA_LOCATION_MEMORY) {
    std::istringstream s;
    s.rdbuf()->pubsetbuf(
        const_cast<char*>(reinterpret_cast<const char*>(location.location)),
        location.size);
    comp->module_ = torch::jit::load(s, c10::Device(g_device.device_t_));
  } else {
    comp->module_ =
        torch::jit::load(reinterpret_cast<const char*>(location.location),
                         c10::Device(g_device.device_t_));
  }
  auto schema = comp->module_.get_method("forward").function().getSchema();
  assert(!schema.is_vararg());
  assert(!schema.is_varret());
  auto num_inputs = comp->module_.get_method("forward").function().num_inputs();
  comp->num_inputs_ = num_inputs - 1;
  for (uint32_t idx = 0; idx < std::max(comp->num_inputs_, MAX_OUTPUT_TENSORS);
       ++idx) {
    auto v = std::make_unique<_odla_value>(idx);
    comp->odla_inputs_outputs_.push_back(v.get());
  }
  *computation = comp.get();
  g_executables[*computation] = std::move(comp);
  return ODLA_SUCCESS;
}

odla_status odla_GetArgFromExecutableByIdx(odla_executable comp,
                                           odla_uint32 idx, odla_value* value) {
  if (idx > comp->num_inputs_) {
    *value = nullptr;
    return ODLA_FAILURE;
  }
  *value = comp->odla_inputs_outputs_[idx];
  return ODLA_SUCCESS;
}

odla_status odla_GetOutputFromExecutableByIdx(const odla_executable comp,
                                              const odla_uint32 output_idx,
                                              odla_value* output_value) {
  if (output_idx > comp->odla_inputs_outputs_.size()) {
    *output_value = nullptr;
    return ODLA_FAILURE;
  }
  *output_value = comp->odla_inputs_outputs_[output_idx];
  return ODLA_SUCCESS;
}

odla_status odla_CreateContext(odla_context* context) {
  *context = nullptr;
  auto ctx = std::make_unique<_odla_context>();
  *context = ctx.get();
  g_ctxs[*context] = std::move(ctx);
  return ODLA_SUCCESS;
}

odla_status odla_SetRuntimeValueType(odla_context context, odla_value v,
                                     odla_value_type ty) {
  assert(v->id_ < MAX_INPUT_TENSORS);
  context->input_types_[v->id_] = std::move(ty);
  return ODLA_SUCCESS;
}

odla_status odla_GetRuntimeValueType(odla_context context, odla_value value,
                                     odla_value_type* ty) {
  assert(value->id_ <= context->num_output_tensors_);
  auto t = context->output_tensors_[value->id_];
  *ty = toODLAValueType(t.scalar_type(), t.sizes());
  return ODLA_SUCCESS;
}

odla_status odla_BindToArgument(odla_value value, const odla_void* data_ptr,
                                odla_context context) {
  assert(value->id_ < MAX_INPUT_TENSORS);
  auto ty = context->input_types_[value->id_];
  auto options = c10::TensorOptions()
                     .dtype(toTensorDataType(ty.element_type))
                     .device(c10::kCPU);
  auto t = at::from_blob(const_cast<void*>(data_ptr), toTensorDim(ty.shape),
                         options);
  if (g_device.device_t_ == c10::kCUDA) {
    t = t.to(c10::device(c10::kCUDA));
  }
  context->inputs_[value->id_] = c10::IValue(t);
  return ODLA_SUCCESS;
}

odla_status odla_BindToOutput(odla_value value, odla_void* data_ptr,
                              odla_context context) {
  assert(value->id_ < context->num_output_tensors_);
  auto t = context->output_tensors_[value->id_];
  auto ty = toODLAValueType(t.scalar_type(), t.sizes());
  void* raw_data = t.storage().data();
  int len = at::elementSize(t.scalar_type()) * getElementCount(ty.shape);
  if (g_device.device_t_ == c10::kCPU) {
    memcpy(data_ptr, raw_data, len);
  } else {
    // cudaMemcpy(data_ptr, raw_data, len, cudaMemcpyDeviceToHost);
    t = t.to(c10::Device(c10::kCPU));
    memcpy(data_ptr, t.storage().data(), len);
  }
  return ODLA_SUCCESS;
}

odla_status odla_GetRuntimeNumOfOutputs(odla_context context,
                                        odla_uint32* num_output_ptr) {
  *num_output_ptr = (odla_uint32)context->num_output_tensors_;
  return ODLA_SUCCESS;
}

odla_status odla_LaunchExecutable(const odla_executable computation,
                                  const odla_context context) {
  context->inputs_.resize(computation->num_inputs_);
  context->input_types_.resize(computation->num_inputs_);
  context->output_ = computation->module_.forward(context->inputs_);

  if (context->output_.isTensor()) {
    context->output_tensors_.push_back(context->output_.toTensor());
  } else {
    assert(context->output_.isTuple());
    for (const auto& item : context->output_.toTuple()->elements()) {
      assert(item.isTensor());
      context->output_tensors_.push_back(item.toTensor());
    }
  }
  context->num_output_tensors_ = context->output_tensors_.size();
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

odla_status odla_DestroyExecutable(odla_executable computation) {
  auto it = g_executables.find(computation);
  if (it == g_executables.end()) {
    return ODLA_FAILURE;
  }
  g_executables.erase(it);
  return ODLA_SUCCESS;
}

odla_status odla_DestroyDevice(odla_device device) { return ODLA_SUCCESS; }