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
#include <stdlib.h>

#include <cstdlib>
#include <fstream>
#include <mutex>
#include <popart/builder.hpp>
#include <popart/devicemanager.hpp>
#include <popart/ndarraywrapper.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/session.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/voiddata.hpp>
#include <poplar/exceptions.hpp>
#include <random>
#include <stdexcept>
#include <string>

#include "ODLA/odla_common.h"
#include "common.h"
#include "odla_pipeline.h"
#include "odla_popart.h"
#include "popart_config.h"

#if !defined(ODLA_VERSION_NUMBER) || (ODLA_VERSION_NUMBER < 50)
#error This library requires minimum ODLA version 0.5
#endif

static std::mutex
    g_computation_mutex; // to lock the create & destroy computation

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
    case 1001: // load cache directly, need set path of cache file
    {
      popart::logging::info("set load_or_save_cache");
      PopartConfig::instance()->set_load_or_save_cache(true);
      PopartConfig::instance()->set_cache_path(
          (std::string) reinterpret_cast<char*>(value));
    } break;
    case 1002:
      setenv("POPART_LOG_LEVEL", "INFO", 1);
    default:
      popart::logging::err("Unsupported property type: {}", type);
      return ODLA_UNSUPPORTED_DATATYPE;
  }
  return ODLA_SUCCESS;
}

odla_status odla_SetContextItem(odla_context context, odla_item_type type,
                                odla_item_value value) {
  if (context == nullptr) {
    return ODLA_INVALID_PARAM;
  }
  switch (type) {
    case ODLA_ASYNC_CALLBACK_FUNC:
      context->async_callback_func =
          reinterpret_cast<int (*)(void*, odla_status)>(value);
      break;
    case ODLA_ASYNC_CALLBACK_ARG:
      context->async_callback_arg = reinterpret_cast<void*>(value);
      break;
    default:
      popart::logging::err("Unsupported context property type: {}", type);
      return ODLA_UNSUPPORTED_DATATYPE;
  }
  return ODLA_SUCCESS;
}

odla_status odla_CreateExecutable(odla_executable* executable,
                                  odla_context context, odla_computation comp) {
  popart::logging::info("Start to create Executable...");
  if (comp == nullptr) {
    popart::logging::err(
        "Failed to create Executable... Computation haven't been intialized.");
    return ODLA_FAILURE;
  } else {
    if (comp->session) {
      popart::logging::info("Create cache file from exist session");
      return comp->compile_and_export();
    } else {
      popart::logging::info("Computation is not initialized. init it first");
      odla_status ret =
          _odla_computation::instance()->init(true); // set is_compile to true
                                                     // this comp init will
                                                     // create executable
      if (ret != ODLA_SUCCESS) {
        popart::logging::err("Failed to init computation when compiling.");
        return ret;
      }
      _odla_computation::instance()->compile_and_export();
    }
  }
  return ODLA_SUCCESS;
}

odla_status odla_StoreExecutable(const odla_char* file_name,
                                 odla_executable executable) {
  return ODLA_SUCCESS;
}

odla_status odla_LoadExecutable(const odla_char* file_name, odla_device device,
                                odla_executable* executable,
                                odla_context* context,
                                odla_computation* computation) {
  return ODLA_SUCCESS;
}

odla_status odla_CreateComputation(odla_computation* comp) {
  std::lock_guard<std::mutex> guard(g_computation_mutex);
  auto injector = PopartConfig::instance()->temp_get_error_inject_env();
  if (injector.empty()) {
    popart::logging::warn("NO VALUE SET for error injector");
    if (nullptr == getenv("POPART_LOG_LEVEL")) { // if POPART_LOG_LEVEL not set
      popart::logging::warn("popart logging level set to warn.");
      popart::logging::setLogLevel(popart::logging::Module::popart,
                                   popart::logging::Level::Warn);
    }
  } else {
    setenv("POPLAR_ENGINE_OPTIONS", injector.c_str(), 1);
    if (nullptr == getenv("POPART_LOG_LEVEL")) { // if POPART_LOG_LEVEL not set
      popart::logging::warn("popart logging level set to info.");
      popart::logging::setLogLevel(popart::logging::Module::popart,
                                   popart::logging::Level::Info);
    }
  }
  if (nullptr != getenv("POPLAR_ENGINE_OPTIONS"))
    popart::logging::info("The env POPLAR_ENGINE_OPTIONS value is: {}",
                          getenv("POPLAR_ENGINE_OPTIONS"));
  else
    popart::logging::info("The env POPLAR_ENGINE_OPTIONS value is not set");

  static void* custom_op_handle = nullptr;
  *comp = _odla_computation::instance();
  popart::logging::warn("computation created: {}",
                        _odla_computation::instance());
  if (custom_op_handle == nullptr) {
    custom_op_handle = dlopen("libcustom_ops.so", RTLD_NOW | RTLD_GLOBAL);
    if (custom_op_handle == nullptr) {
      assert(0);
      return ODLA_DL_ERROR;
    }
  }
  // Read the config file
  popart::logging::info("loading config");
  if (!PopartConfig::instance()->inited()) {
    auto ret = PopartConfig::instance()->load_config(
        std::getenv("ODLA_POPART_CONFIG"));
    if (ret != ODLA_SUCCESS) {
      popart::logging::err("error load config");
      return ret;
    }
  }
  odla_status status = _odla_computation::instance()->set_executor();
  if (status != ODLA_SUCCESS) {
    popart::logging::err("set_executor failed");
    return ODLA_FAILURE;
  }
  if (PopartConfig::instance()->execution_mode() == PARALLEL ||
      PopartConfig::instance()->execution_mode() == PIPELINE ||
      PopartConfig::instance()->execution_mode() == PIPELINE_ASYNC) {
    QManager::instance()->createQ(PopartConfig::instance()->queue_type());
    QManager::instance()->getQ()->init(
        PopartConfig::instance()->queue_capacity());
  }

  return ODLA_SUCCESS;
}

odla_status odla_CreateContext(odla_context* context) {
  if (PopartConfig::instance()->execution_mode() == PIPELINE_ASYNC)
    *context = new _odla_pipeline_async_context(_odla_computation::instance());
  else if (PopartConfig::instance()->execution_mode() == UNKNOWN) {
    popart::logging::info(
        "[VODLA DEBUG] Got UNKNOWN execution mode: {},  when "
        "create context, use async as it created before set execution mode",
        PopartConfig::instance()->execution_mode());
    *context = new _odla_pipeline_async_context(_odla_computation::instance());
  } else {
    popart::logging::info(
        "[VODLA DEBUG] Got not PIPELIINE_ASYNC or UNKNOWN execution mode: {},  "
        "when "
        "create context, use pipeline context",
        PopartConfig::instance()->execution_mode());
    *context = new _odla_pipeline_context(_odla_computation::instance());
  }
  return ODLA_SUCCESS;
}

odla_status odla_DestroyContext(odla_context ctx) {
  if (nullptr != ctx && ctx->hold("odla_DestroyContext")) delete (ctx);
  return ODLA_SUCCESS;
}

odla_status odla_DestroyComputation(odla_computation comp) {
  std::lock_guard<std::mutex> guard(g_computation_mutex);
  popart::logging::warn("call odla_destroyComputation comp: {}", comp);
  if (comp != nullptr) {
    if (!comp->is_compile_only()) {
      comp->mark_done();
      QManager::instance()->deleteQ(); // delete current queue
    }
    comp->release_session();
    _odla_computation::destruct(); // release the real computation
  }
  popart::logging::warn("reset config state, comp: {}", comp);
  PopartConfig::instance()->reset_init_state();

  return ODLA_SUCCESS;
}

odla_status odla_ExecuteComputation(odla_computation comp, odla_context context,
                                    odla_compute_mode mode,
                                    odla_device device) {
  if (_odla_computation::instance()->is_compile_only()) {
    popart::logging::err(
        "This computation is created for compile executable, please re-create "
        "another computation for computing");
    return ODLA_FAILURE;
  }
  if (!context->hold("odla_ExecuteComputation")) {
    return ODLA_FAILURE;
  }
  return comp->executor()->compute(comp, context, mode, device);
}

odla_status odla_AsyncExecuteComputation(odla_computation comp,
                                         odla_context context,
                                         odla_compute_mode mode,
                                         odla_device device) {
  odla_status s = ODLA_FAILURE;
  // check the execution mode is PIPELINE_ASYNC or not
  if (PIPELINE_ASYNC != PopartConfig::instance()->execution_mode()) {
    popart::logging::err(
        "execution mode: was {} mismatched when try to run "
        "odla_AsyncExecuteComputation, please set execution_mode to "
        "pipeline_async in ODLA_POPART_CONFIG",
        PopartConfig::instance()->execution_mode());
  } else
    s = odla_ExecuteComputation(comp, context, mode, device);
  if (ODLA_SUCCESS != s && nullptr != context) {
    popart::logging::err(
        "odla_ExecuteComputation in async with context: {} got return status: "
        "{}",
        context, s);
    if (context->async_callback_func && context->async_callback_arg) {
      popart::logging::err("callback to notify the error for ctx: {}", context);
      context->async_callback_func(context->async_callback_arg, s);
    } else
      popart::logging::err(
          "Can't notify the failure status: {} as null callback func:{} or "
          "arg:{}",
          s, context->async_callback_func, context->async_callback_arg);
  }
  return s;
}

odla_value odla_CreateArgument(odla_value_type type, const odla_value_id id) {
  const auto& name = id ? std::string(reinterpret_cast<const char*>(id)) : "";
  popart::TensorInfo tensor_info(GetPopartType(type),
                                 GetPopartShape(type.shape));
  auto comp = _odla_computation::instance();
  popart::TensorId tensor_id = comp->builder->addInputTensor(tensor_info, name);
  auto v = new _odla_value(tensor_id, tensor_info, name);
  comp->inputs_map[name] = v;
  comp->input_values.push_back(v);
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
    return ODLA_INVALID_PARAM;
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
      _odla_computation::instance()->builder->aiOnnxOpset10().constant(data,
                                                                       name);
  return new _odla_value(tensor_id, tensor_info, name);
}

odla_status odla_BindToArgument(odla_value value, const odla_void* data_ptr,
                                odla_context context) {
  odla_status status =
      _odla_computation::instance(false)
          ->init(); // to avoid long execution and async context problem
  if (status != ODLA_SUCCESS &&
      _odla_computation::instance()->session == nullptr) {
    popart::logging::err("init computation item in CreateContext failed.");
    if (PopartConfig::instance()->execution_mode() == PIPELINE_ASYNC &&
        nullptr != context) {
      popart::logging::err(
          "odla_ExecuteComputation in async with context: {} got return "
          "status: {}",
          context, status);
      if (context->async_callback_func && context->async_callback_arg) {
        popart::logging::err("callback to notify the error for ctx: {}",
                             context);
        context->async_callback_func(context->async_callback_arg, status);
      } else
        popart::logging::err(
            "Can't notify the failure status: {} as null callback func:{} or "
            "arg:{}",
            status, context->async_callback_func, context->async_callback_arg);
    }
    return status;
  }
  if (!context->hold("odla_BindToArgument")) return ODLA_FAILURE;
  std::vector<int64_t> shape =
      context->comp->builder->getTensorShape(value->tensor_id);
  // only the SEQUENCE model need to pass the data in once time
  if (PopartConfig::instance()->execution_mode() == SEQUENCE &&
      shape.size() > 0)
    shape[0] *= g_comp->opts.batches_per_step;
  std::unique_ptr<popart::IArray> p_array = MakeNDArrayWrapper(
      data_ptr, context->comp->builder->getTensorDataType(value->tensor_id),
      shape);
  popart::logging::info("Bind the value to input {}", value->tensor_id);
  context->inputs[value->tensor_id] = std::move(p_array);
  return ODLA_SUCCESS;
}

odla_status odla_BindToArgumentById(const odla_value_id value_id,
                                    const odla_void* data_ptr,
                                    odla_context context) {
  if (!context->hold("odla_BindToArgumentById")) return ODLA_FAILURE;
  std::string name(reinterpret_cast<const char*>(value_id));
  return odla_BindToArgument(context->comp->inputs_map[name], data_ptr,
                             context);
}

odla_status odla_SetValueAsOutput(const odla_value value) {
  auto comp = _odla_computation::instance();
  comp->builder->addOutputTensor(value->tensor_id);
  comp->outputs_map[value->name] = value;
  comp->output_values.push_back(value);
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
    return ODLA_INVALID_PARAM;
  }
  *output_value = computation->output_values[output_idx];
  return ODLA_SUCCESS;
}

odla_status odla_BindToOutput(odla_value value, odla_void* data_ptr,
                              odla_context context) {
  if (!context->hold("odla_BindToOutput")) return ODLA_FAILURE;
  std::vector<int64_t> shape =
      context->comp->builder->getTensorShape(value->tensor_id);
  // only the SEQUENCE model need to pass the data in once time
  if (PopartConfig::instance()->execution_mode() == SEQUENCE &&
      shape.size() > 0)
    shape[0] *= g_comp->opts.batches_per_step;
  std::unique_ptr<popart::IArray> p_array = MakeNDArrayWrapper(
      data_ptr, context->comp->builder->getTensorDataType(value->tensor_id),
      shape);
  context->outputs[value->tensor_id] = std::move(p_array);
  return ODLA_SUCCESS;
}

odla_status odla_BindToOutputById(const odla_value_id value_id,
                                  odla_void* data_ptr, odla_context context) {
  if (!context->hold("odla_BindToOutputById")) return ODLA_FAILURE;
  std::string name(reinterpret_cast<const char*>(value_id));
  return odla_BindToOutput(context->comp->outputs_map[name], data_ptr, context);
}

odla_status odla_GetValueType(const odla_value value,
                              odla_value_type* value_type) {
  value_type->element_type = GetOdlaType(value->tensor_info.dataType());
  value_type->shape = GetOdlaShape(value->tensor_info.shape());
  return ODLA_SUCCESS;
}
