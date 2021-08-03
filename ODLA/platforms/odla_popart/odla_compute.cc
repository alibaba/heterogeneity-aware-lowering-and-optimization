//===- odla_compute.cc ----------------------------------------------------===//
//
// Copyright (C) 2019-2020 Alibaba Group Holding Limited.
// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
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
#include <dlfcn.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <functional>
#include <memory>
#include <numeric>
#include <popart/builder.hpp>
#include <popart/devicemanager.hpp>
#include <popart/names.hpp>
#include <popart/ndarraywrapper.hpp>
#include <popart/session.hpp>
#include <popart/sessionoptions.hpp>
#include <popart/stepio.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/voiddata.hpp>
#include <popart/names.hpp>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <mutex>
#include <queue>
#include <thread>
#include <array>
#include <fstream>
#include <sstream>

#include "ODLA/odla_common.h"
#include "common.h"
#include "odla_popart.h"
#include "odla_pipeline.h"
#include "popart_config.h"

#if !defined(ODLA_VERSION_NUMBER) || (ODLA_VERSION_NUMBER < 50)
#error This library requires minimum ODLA version 0.5
#endif

PopartConfig* PopartConfig::m_instance = new PopartConfig();

odla_status odla_SetComputationItem(odla_computation comp, odla_item_type type,
                                    odla_item_value value) {
  std::cout << "---> odla_SetComputationItem()" << std::endl;
  switch (type) {
    case ODLA_USE_SIM_MODE:
      comp->opts.use_ipu_model = *(reinterpret_cast<bool*>(value));
      break;
    case ODLA_PROCESSOR_NUM:
      comp->opts.ipu_num = *(reinterpret_cast<int*>(value));
      break;
    case ODLA_BATCHES_PER_STEP:
      comp->opts.batches_per_step = *(reinterpret_cast<int*>(value));
      break;
    default:
      std::cerr << "Unsupported property type: " << type << std::endl;
      return ODLA_FAILURE;
  }
  std::cout << "<--- odla_SetComputationItem()" << std::endl;
  return ODLA_SUCCESS;
}

odla_status odla_CreateComputation(odla_computation* comp) {
  std::cout << "---> odla_CreateComputation()" << std::endl;
  static void* custom_op_handle = nullptr;
  *comp = _odla_computation::instance();
  if (custom_op_handle == nullptr) {
    custom_op_handle = dlopen("libcustom_ops.so", RTLD_NOW | RTLD_GLOBAL);
    if (custom_op_handle == nullptr) {
      std::cerr << "Unable to open libcustom_ops " << dlerror() << std::endl;
      assert(0);
      return ODLA_FAILURE;
    }
  }
  //Read the config file
  PopartConfig::instance()->load_config("Please_write_test_parameter_in_it");
  _odla_computation::instance()->set_executor();
  std::cout << "<--- odla_CreateComputation()" << std::endl;
  return ODLA_SUCCESS;
}

odla_status odla_CreateContext(odla_context* context) {
  std::cout << "---> odla_CreateContext()" << std::endl;
  *context = new _odla_pipeline(_odla_computation::instance());
  std::cout << "<--- odla_CreateContext()" << std::endl;
  return ODLA_SUCCESS;
}

odla_status odla_DestroyContext(odla_context ctx) {
  std::cout << "---> odla_DestroyContext()" << std::endl;
  if(nullptr != ctx)
    delete (ctx);
  else
    std::cerr << "Encounter a odla_DestroyContext with null ctx" << std::endl;
  std::cout << "<--- odla_DestroyContext()" << std::endl;
  return ODLA_SUCCESS;
}

odla_status odla_DestroyComputation(odla_computation comp) {
  std::cout << "Mark the computation done ..." << std::endl;
  comp->mark_done();
  return ODLA_SUCCESS;
}

odla_status odla_ExecuteComputation(odla_computation comp, odla_context context,
                                    odla_compute_mode mode,
                                    odla_device device) {
  std::cout << "---> odla_ExecuteComputation()" << std::endl;
  comp->executor()->compute(comp, context, mode, device);
  std::cout << "<--- odla_ExecuteComputation()" << std::endl;
  return ODLA_SUCCESS;
}

odla_value odla_CreateArgument(odla_value_type type, const odla_value_id id) {
  std::cout << "---> odla_CreateArgument()" << std::endl;
  const auto& name = id ? std::string(reinterpret_cast<const char*>(id)) : "";
  popart::TensorInfo tensor_info(GetPopartType(type),
                                 GetPopartShape(type.shape));
  auto comp = _odla_computation::instance();
  popart::TensorId tensor_id =
      comp->builder->addInputTensor(tensor_info, name);
  auto v = new _odla_value(tensor_id, tensor_info, name);
  comp->inputs_map[name] = v;
  comp->input_values.push_back(v);
  std::cout << "<--- odla_CreateArgument()" << std::endl;
  return v;
}

odla_status odla_GetNumOfArgsFromComputation(const odla_computation computation,
                                             odla_uint32* num_args) {
  std::cout << "---> odla_GetNumOfArgsFromComputation()" << std::endl;
  *num_args = computation->input_values.size();
  std::cout << "<--- odla_GetNumOfArgsFromComputation()" << std::endl;
  return ODLA_SUCCESS;
}

odla_status odla_GetArgFromComputationByIdx(const odla_computation computation,
                                            const odla_uint32 arg_idx,
                                            odla_value* arg_value) {
  std::cout << "---> odla_GetArgFromComputationByIdx()" << std::endl;
  *arg_value = nullptr;
  if (arg_idx >= computation->input_values.size()) {
    return ODLA_FAILURE;
  }
  *arg_value = computation->input_values[arg_idx];
  std::cout << "<--- odla_GetArgFromComputationByIdx()" << std::endl;
  return ODLA_SUCCESS;
}

odla_value odla_CreateConstant(odla_value_type type, const void* data_ptr,
                               const odla_value_id id) {
  std::cout << "---> odla_CreateConstant()" << std::endl;
  const auto& name = id ? std::string(reinterpret_cast<const char*>(id)) : "";
  popart::TensorInfo tensor_info(GetPopartType(type),
                                 GetPopartShape(type.shape));
  popart::ConstVoidData data = {
      data_ptr, {GetPopartType(type), GetPopartShape(type.shape)}};
  popart::TensorId tensor_id =
      _odla_computation::instance()->builder->aiOnnxOpset10().constant(data, name);
  std::cout << "<--- odla_CreateConstant()" << std::endl;
  return new _odla_value(tensor_id, tensor_info, name);
}

odla_status odla_BindToArgument(odla_value value, const odla_void* data_ptr,
                                odla_context context) {
  std::cout << "---> odla_BindToArgument() : " << context << std::endl;
  std::unique_ptr<popart::IArray> p_array = MakeNDArrayWrapper(
      data_ptr, context->comp->builder->getTensorDataType(value->tensor_id),
      context->comp->builder->getTensorShape(value->tensor_id));
  context->inputs[value->tensor_id] = std::move(p_array);
  std::cout << "<--- odla_BindToArgument()" << std::endl;
  return ODLA_SUCCESS;
}

odla_status odla_BindToArgumentById(const odla_value_id value_id,
                                    const odla_void* data_ptr,
                                    odla_context context) {
  std::cout << "---> odla_BindToArgumentById() : " << context << std::endl;
  std::string name(reinterpret_cast<const char*>(value_id));
  std::cout << "<--- odla_BindToArgumentById()" << std::endl;
  return odla_BindToArgument(context->comp->inputs_map[name], data_ptr,
                             context);
}

odla_status odla_SetValueAsOutput(const odla_value value) {
  std::cout << "---> odla_SetValueAsOutput()" << std::endl;
  auto comp = _odla_computation::instance();
  comp->builder->addOutputTensor(value->tensor_id);
  comp->outputs_map[value->name] = value;
  comp->output_values.push_back(value);
  std::cout << "<--- odla_SetValueAsOutput()" << std::endl;
  return ODLA_SUCCESS;
}

odla_status odla_SetValuesAsOutput(const odla_values values) {
  std::cout << "---> odla_SetValuesAsOutput()" << std::endl;
  for (int i = 0; i < values.size; ++i) {
    odla_SetValueAsOutput(values.values[i]);
  }
  std::cout << "<--- odla_SetValuesAsOutput()" << std::endl;
  return ODLA_SUCCESS;
}

odla_status odla_GetNumOfOutputsFromComputation(
    const odla_computation computation, odla_uint32* num_outputs) {
  std::cout << "---> odla_GetNumOfOutputsFromComputation()" << std::endl;
  *num_outputs = computation->output_values.size();
  std::cout << "<--- odla_GetNumOfOutputsFromComputation()" << std::endl;
  return ODLA_SUCCESS;
}

odla_status odla_GetOutputFromComputationByIdx(
    const odla_computation computation, const odla_uint32 output_idx,
    odla_value* output_value) {
  std::cout << "---> odla_GetOutputFromComputationByIdx()" << std::endl;
  *output_value = nullptr;
  if (output_idx >= computation->output_values.size()) {
    return ODLA_FAILURE;
  }
  *output_value = computation->output_values[output_idx];
  std::cout << "<--- odla_GetOutputFromComputationByIdx()" << std::endl;
  return ODLA_SUCCESS;
}

odla_status odla_BindToOutput(odla_value value, odla_void* data_ptr,
                              odla_context context) {
  std::cout << "---> odla_BindToOutput()" << std::endl;
  std::unique_ptr<popart::IArray> p_array = MakeNDArrayWrapper(
      data_ptr, context->comp->builder->getTensorDataType(value->tensor_id),
      context->comp->builder->getTensorShape(value->tensor_id));
  context->outputs[value->tensor_id] = std::move(p_array);
  std::cout << "<--- odla_BindToOutput()" << std::endl;
  return ODLA_SUCCESS;
}

odla_status odla_BindToOutputById(const odla_value_id value_id,
                                  odla_void* data_ptr, odla_context context) {
  std::cout << "---> odla_BindToOutputById()" << std::endl;
  std::string name(reinterpret_cast<const char*>(value_id));
  return odla_BindToOutput(context->comp->outputs_map[name], data_ptr, context);
  std::cout << "<--- odla_BindToOutputById()" << std::endl;
}

odla_status odla_GetValueType(const odla_value value,
                              odla_value_type* value_type) {
  std::cout << "---> odla_GetValueType()" << std::endl;
  value_type->element_type = GetOdlaType(value->tensor_info.dataType());
  value_type->shape = GetOdlaShape(value->tensor_info.shape());
  std::cout << "<--- odla_GetValueType()" << std::endl;
  return ODLA_SUCCESS;
}