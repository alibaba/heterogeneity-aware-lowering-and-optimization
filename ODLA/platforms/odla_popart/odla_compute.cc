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
#include <popart/dataflow.hpp>
#include <popart/devicemanager.hpp>
#include <popart/names.hpp>
#include <popart/ndarraywrapper.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/session.hpp>
#include <popart/sessionoptions.hpp>
#include <popart/stepio.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/voiddata.hpp>
#include <poplar/exceptions.hpp>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "ODLA/odla_common.h"
#include "common.h"
#include "odla_popart.h"

#if !defined(ODLA_VERSION_NUMBER) || (ODLA_VERSION_NUMBER < 50)
#error This library requires minimum ODLA version 0.5
#endif

thread_local odla_computation g_comp;
static std::vector<std::unique_ptr<_odla_computation>> g_comps;

static std::shared_ptr<popart::DeviceInfo> AcquireAvailableDevice(
    int num_devices) {
  return popart::DeviceManager::createDeviceManager().acquireAvailableDevice(
      num_devices);
}

static std::shared_ptr<popart::DeviceInfo> CreateIpuModelDevice(
    int num_devices) {
  std::map<std::string, std::string> deviceOpts{
      {"numIPUs", std::to_string(num_devices)}, {"tilesPerIPU", "1216"}};
  return popart::DeviceManager::createDeviceManager().createIpuModelDevice(
      deviceOpts);
}

std::unique_ptr<popart::SessionOptions> SessionOptions() {
  auto opts =
      std::unique_ptr<popart::SessionOptions>(new popart::SessionOptions());
  opts->virtualGraphMode = popart::VirtualGraphMode::Auto;
  opts->enableStochasticRounding = true;
  const char* envEngineCachePath = getenv("ENGINE_CACHE_PATH");
  if (g_comp->opts.enable_engine_cache || envEngineCachePath != nullptr) {
    opts->enableEngineCaching = true;
    opts->cachePath = g_comp->opts.enable_engine_cache ? g_comp->opts.cache_dir
                                                       : envEngineCachePath;
  }
  return opts;
}

odla_status odla_SetComputationItem(odla_computation comp, odla_item_type type,
                                    odla_item_value value) {
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
    case ODLA_ENABLE_ENGINE_CACHE:
      comp->opts.enable_engine_cache = *(reinterpret_cast<bool*>(value));
      break;
    case ODLA_CACHE_DIR:
      comp->opts.cache_dir = (reinterpret_cast<char*>(value));
      break;
    default:
      std::cerr << "Unsupported property type: " << type << std::endl;
      return ODLA_FAILURE;
  }

  return ODLA_SUCCESS;
}

odla_status odla_StoreExecutable(const odla_char* file_name,
                                 odla_executable executable) {
  return ODLA_SUCCESS;
}

odla_status odla_LoadExecutable(const odla_char* file_name,
                                odla_executable* executable,
                                odla_context* context,
                                odla_computation* computation) {
  return ODLA_SUCCESS;
}

odla_status odla_CreateComputation(odla_computation* comp) {
  // Create graph builder
  static void* custom_op_handle = nullptr;
  std::unique_ptr<popart::Builder> builder = popart::Builder::create();

  // Place Subgraph on IPU 0
  builder->setAttribute(popart::sVirtualGraphAttribute, 0);
  // TODO(unknown) support shard mode
  // builder->virtualGraph(inst_tensor_id, 0/*device id*/);
  g_comps.push_back(std::unique_ptr<_odla_computation>(
      new _odla_computation(std::move(builder))));
  g_comp = g_comps.back().get();
  *comp = g_comp;
  if (custom_op_handle == nullptr) {
    custom_op_handle = dlopen("libcustom_ops.so", RTLD_NOW | RTLD_GLOBAL);
    if (custom_op_handle == nullptr) {
      std::cerr << "Unable to open libcustom_ops " << dlerror() << std::endl;
      assert(0);
      return ODLA_FAILURE;
    }
  }

  return ODLA_SUCCESS;
}

odla_status odla_CreateContext(odla_context* context) {
  *context = new _odla_context(g_comp);
  // Create dataflow
  std::vector<popart::TensorId> ids;
  for (const auto& output : g_comp->outputs_map) {
    ids.push_back(output.second->tensor_id);
  }

  // Batches per step is a compile time constant value
  popart::DataFlow data_flow(g_comp->opts.batches_per_step, ids,
                             popart::AnchorReturnType("All"));

  // Acquire IPU
  auto device = g_comp->opts.use_ipu_model
                    ? CreateIpuModelDevice(g_comp->opts.ipu_num)
                    : AcquireAvailableDevice(g_comp->opts.ipu_num);

  // Create and config SessionOptions
  auto opts = SessionOptions();

  // Create InferenceSession
  auto proto = g_comp->builder->getModelProto();
  g_comp->session = popart::InferenceSession::createFromOnnxModel(
      proto, data_flow, device, popart::InputShapeInfo(), *opts);

  // Compile graph, create engine and load into the IPU
  // use compileAndExport() to frozen engine to specified path
  g_comp->session->prepareDevice();
  // Init seed
  g_comp->session->setRandomSeed(0);
  // Copy weights from host to IPU
  g_comp->session->weightsFromHost();

  return ODLA_SUCCESS;
}

odla_status odla_DestroyContext(odla_context ctx) {
  if (ctx != nullptr) {
    delete (ctx);
  }
  return ODLA_SUCCESS;
}

odla_status odla_DestroyComputation(odla_computation comp) {
  if (comp->session != nullptr) {
    comp->session->getDevice().getDeviceInfo()->detach();
    comp->session.reset();
  }

  if (g_comp == comp && g_comp != nullptr) {
    g_comp = nullptr;
  }

  return ODLA_SUCCESS;
}

odla_status odla_ExecuteComputation(odla_computation comp, odla_context context,
                                    odla_compute_mode mode,
                                    odla_device device) {
  // Config StepIO
  std::map<popart::TensorId, popart::IArray&> inputs;
  for (auto& input : context->inputs) {
    inputs.emplace(input.first, *input.second);
  }
  std::map<popart::TensorId, popart::IArray&> outputs;
  for (auto& output : context->outputs) {
    outputs.emplace(output.first, *output.second);
  }

  popart::StepIO stepio(inputs, outputs);
  // Run on ipu
  try {
    g_comp->session->run(stepio);
  }
  catch (poplar::application_runtime_error &e) {
    popart::logging::info("Poplar exception application_runtime_error caught:");
    return INTERNAL_LOGIC_ERR;
  }
  catch (poplar::recoverable_runtime_error &e) {
    popart::logging::info("Poplar recoverable_runtime_error exception caught");
    auto action = e.getRecoveryAction();
    popart::logging::info("need to take action:{}", action);
    if (action == poplar::RecoveryAction::IPU_RESET) {
      return RECOVERABLE_ERR;
    }
    else if (action == poplar::RecoveryAction::PARTITION_RESET) {
      return PARTITION_RESET; 
    }
    else if (action == poplar::RecoveryAction::FULL_RESET) {
      return FULL_RESET;
    }
  }
  catch (poplar::unrecoverable_runtime_error &e) {
    popart::logging::info("Poplar unrecoverable_runtime_error exception caught");
    return UNRECOVERABLE_ERR; 
  }
  catch(poplar::unknown_runtime_error &e) {
    popart::logging::info("Poplar unknown exception caught}");
    return UNRECOVERABLE_ERR;
  }
  catch(...) {
    popart::logging::info("Poplar unknown exception caught");
    return UNRECOVERABLE_ERR;
  }

  return ODLA_SUCCESS;
}

odla_value odla_CreateArgument(odla_value_type type, const odla_value_id id) {
  const auto& name = id ? std::string(reinterpret_cast<const char*>(id)) : "";
  popart::TensorInfo tensor_info(GetPopartType(type),
                                 GetPopartShape(type.shape));
  popart::TensorId tensor_id =
      g_comp->builder->addInputTensor(tensor_info, name);
  auto v = new _odla_value(tensor_id, tensor_info, name);
  g_comp->inputs_map[name] = v;
  g_comp->input_values.push_back(v);
  return v;
}

odla_status odla_GetNumOfArgsFromComputation(const odla_computation computation,
                                             odla_uint32* num_args) {
  *num_args = computation->input_values.size();
  return ODLA_SUCCESS;
}

odla_status odla_GetArgFromComputationByIdx(const odla_computation computation,
                                            const odla_uint32 arg_idx,
                                            odla_value* arg_value) {
  *arg_value = nullptr;
  if (arg_idx >= computation->input_values.size()) {
    return ODLA_FAILURE;
  }
  *arg_value = computation->input_values[arg_idx];
  return ODLA_SUCCESS;
}

odla_value odla_CreateConstant(odla_value_type type, const void* data_ptr,
                               const odla_value_id id) {
  const auto& name = id ? std::string(reinterpret_cast<const char*>(id)) : "";
  popart::TensorInfo tensor_info(GetPopartType(type),
                                 GetPopartShape(type.shape));
  popart::ConstVoidData data = {
      data_ptr, {GetPopartType(type), GetPopartShape(type.shape)}};
  popart::TensorId tensor_id =
      g_comp->builder->aiOnnxOpset10().constant(data, name);
  return new _odla_value(tensor_id, tensor_info, name);
}

odla_status odla_BindToArgument(odla_value value, const odla_void* data_ptr,
                                odla_context context) {
  std::unique_ptr<popart::IArray> p_array = MakeNDArrayWrapper(
      data_ptr, context->comp->builder->getTensorDataType(value->tensor_id),
      context->comp->builder->getTensorShape(value->tensor_id));
  context->inputs[value->tensor_id] = std::move(p_array);
  return ODLA_SUCCESS;
}

odla_status odla_BindToArgumentById(const odla_value_id value_id,
                                    const odla_void* data_ptr,
                                    odla_context context) {
  std::string name(reinterpret_cast<const char*>(value_id));
  return odla_BindToArgument(g_comp->inputs_map[name], data_ptr, context);
}

odla_status odla_SetValueAsOutput(const odla_value value) {
  g_comp->builder->addOutputTensor(value->tensor_id);
  g_comp->outputs_map[value->name] = value;
  g_comp->output_values.push_back(value);
  return ODLA_SUCCESS;
}

odla_status odla_SetValuesAsOutput(const odla_values values) {
  for (int i = 0; i < values.size; ++i) {
    odla_SetValueAsOutput(values.values[i]);
  }
  return ODLA_SUCCESS;
}

odla_status odla_GetNumOfOutputsFromComputation(
    const odla_computation computation, odla_uint32* num_outputs) {
  *num_outputs = computation->output_values.size();
  return ODLA_SUCCESS;
}

odla_status odla_GetOutputFromComputationByIdx(
    const odla_computation computation, const odla_uint32 output_idx,
    odla_value* output_value) {
  *output_value = nullptr;
  if (output_idx >= computation->output_values.size()) {
    return ODLA_FAILURE;
  }
  *output_value = computation->output_values[output_idx];
  return ODLA_SUCCESS;
}

odla_status odla_BindToOutput(odla_value value, odla_void* data_ptr,
                              odla_context context) {
  std::unique_ptr<popart::IArray> p_array = MakeNDArrayWrapper(
      data_ptr, context->comp->builder->getTensorDataType(value->tensor_id),
      context->comp->builder->getTensorShape(value->tensor_id));
  context->outputs[value->tensor_id] = std::move(p_array);
  return ODLA_SUCCESS;
}

odla_status odla_BindToOutputById(const odla_value_id value_id,
                                  odla_void* data_ptr, odla_context context) {
  std::string name(reinterpret_cast<const char*>(value_id));
  return odla_BindToOutput(g_comp->outputs_map[name], data_ptr, context);
}

odla_status odla_GetValueType(const odla_value value,
                              odla_value_type* value_type) {
  value_type->element_type = GetOdlaType(value->tensor_info.dataType());
  value_type->shape = GetOdlaShape(value->tensor_info.shape());
  return ODLA_SUCCESS;
}
