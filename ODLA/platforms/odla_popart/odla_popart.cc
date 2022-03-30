//===- odla_popart.cc -----------------------------------------------------===//
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
#include "odla_popart.h"

#include <bits/stdc++.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

#include <chrono>
#include <fstream>
#include <iostream>
#include <mutex>
#include <popart/builder.hpp>
#include <popart/dataflow.hpp>
#include <popart/version.hpp>

#include "odla_pipeline.h"
#include "onnx/onnx.pb.h"
#include "popart_config.h"

_odla_computation* _odla_computation::instance_ = nullptr;
std::mutex _odla_computation::comp_mutex_;

#define POPLAR_TRY try {
#define POPLAR_CATCH                                                          \
  }                                                                           \
  catch (poplar::application_runtime_error & e) {                             \
    popart::logging::err(                                                     \
        "Poplar exception application_runtime_error caught: {}", e.what());   \
    RETURN_ERROR(ODLA_INTERNAL_LOGIC_ERR)                                     \
  }                                                                           \
  catch (poplar::recoverable_runtime_error & e) {                             \
    popart::logging::err(                                                     \
        "Poplar recoverable_runtime_error exception caught: {}", e.what());   \
    auto action = e.getRecoveryAction();                                      \
    popart::logging::err("need to take action:{}", action);                   \
    if (action == poplar::RecoveryAction::IPU_RESET) {                        \
      RETURN_ERROR(ODLA_RECOVERABLE_ERR)                                      \
    } else if (action == poplar::RecoveryAction::PARTITION_RESET) {           \
      RETURN_ERROR(ODLA_PARTITION_RESET)                                      \
    } else if (action == poplar::RecoveryAction::FULL_RESET) {                \
      RETURN_ERROR(ODLA_FULL_RESET)                                           \
    }                                                                         \
  }                                                                           \
  catch (poplar::unrecoverable_runtime_error & e) {                           \
    popart::logging::err(                                                     \
        "Poplar unrecoverable_runtime_error exception caught: {}", e.what()); \
    RETURN_ERROR(ODLA_UNRECOVERABLE_ERR)                                      \
  }                                                                           \
  catch (poplar::unknown_runtime_error & e) {                                 \
    popart::logging::err("Poplar unknown runtime exception caught: {}",       \
                         e.what());                                           \
    RETURN_ERROR(ODLA_UNRECOVERABLE_ERR)                                      \
  }                                                                           \
  catch (std::exception & e) {                                                \
    popart::logging::err("std::exception gotten: {}", e.what());              \
    RETURN_ERROR(ODLA_UNRECOVERABLE_ERR)                                      \
  }                                                                           \
  catch (...) {                                                               \
    popart::logging::err("Poplar unknown exception caught");                  \
    RETURN_ERROR(ODLA_UNRECOVERABLE_ERR)                                      \
  }

#define RETURN_ERROR(ERR_CODE) (QManager::instance()->set_status(ERR_CODE));

void compute_loop(odla_computation comp) {
  // setup the stepio with allbacks
  popart::StepIOCallback stepio(input_callback, input_complete_callback,
                                output_callback, output_complete_callback);
  int i = 0;
  bool info_printed = false;
  POPLAR_TRY
  comp->set_thread_run(); // set the state to RUNNING
  while (!comp->is_done()) {
    auto start = std::chrono::steady_clock::now();
    popart::logging::info("This is the {} time for the inference", i++);
    if (i == INT_MAX) i = 0;
    if (!info_printed) {
      popart::logging::warn(
          "Start to run the stepio with comp:{}, session:{}, device:{}", comp,
          comp->session.get(), comp->session->getDevice().getDeviceInfo());
      info_printed = true;
    }
    comp->session->run(stepio);
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    popart::logging::info(
        "[ {} ] ONE_STEP takes {} s. Check whether more inference tasks "
        "wating.",
        i, elapsed_seconds.count());
    // Make wait on CPU if there's not inference task
    start = std::chrono::steady_clock::now();
    while (!comp->is_done() && QManager::instance()->getQ()->size() == 0)
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    end = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> elapsed_ms = end - start;
    popart::logging::info("Found new tasks in {} ms.", elapsed_ms.count());
  }
  POPLAR_CATCH

  popart::logging::warn(
      "The computation: {} pipeline loop finished after {} steps run", comp, i);
  comp->thread_done();
}

#undef RETURN_ERROR
#define RETURN_ERROR(ERR_CODE) return ERR_CODE;

odla_status _odla_computation::compile_and_export() {
  odla_status ret_value = ODLA_SUCCESS;
  POPLAR_TRY
  popart::logging::warn("Start compile and export");
  const std::string& cache_file_name =
      PopartConfig::instance()->get_cache_path();
  std::string file_suffix(".popart");
  int file_prefix = cache_file_name.rfind(file_suffix);
  if (file_prefix == std::string::npos ||
      file_prefix + file_suffix.size() < cache_file_name.size()) {
    popart::logging::err(
        "Bad cache file name. File name should end with '.popart'");
    return ODLA_FAILURE;
  }
  if (file_prefix == std::string::npos) {
    file_prefix = cache_file_name.size() - 1;
  }
  std::string config_file_name(cache_file_name.substr(0, file_prefix) +
                               ".json");
  std::fstream cache_fs(cache_file_name, std::ios_base::out |
                                             std::ifstream::binary |
                                             std::ios_base::trunc);
  if (!cache_fs.is_open()) {
    popart::logging::err("Open or create cache file falied");
    return ODLA_FAILURE;
  }
  std::fstream config_fs;
  std::string config_string;
  if (config_file_name.size() > 0) {
    config_fs.open(config_file_name, std::ios_base::in | std::ifstream::binary);
    if (!config_fs.is_open()) {
      popart::logging::warn(
          "Open config file failed:[ {} ] will use default config",
          config_file_name);
      PopartConfig::instance()->use_default();
      config_string = PopartConfig::instance()->get_default_config_string();
    } else {
      std::ostringstream config_ss;
      config_ss << config_fs.rdbuf();
      config_string = config_ss.str();
    }
  } else {
    config_string = PopartConfig::instance()->get_default_config_string();
  }
  // add sdk_version in the file content
  std::string version_string(popart::core::packageHash());
  popart::logging::info("the popart package hash is: {}", version_string);
  if (config_string.find("sdk_version") == std::string::npos) {
    std::string item_string = "\n\"sdk_version\":\"" + version_string + "\",";
    config_string.insert(1, item_string);
  }
  // change all true to false
  std::string src_replace("true");
  std::string dest_replace("false");
  std::string::size_type pos = 0;
  while ((pos = config_string.find(src_replace)) != std::string::npos)
    config_string.replace(pos, src_replace.length(), dest_replace);
  popart::logging::info("the config_string with sdk_version is: {}",
                        config_string);
  // added the sdk_version information to the file content
  int config_size = config_string.size();
  cache_fs.write((char*)&config_size, sizeof(config_size));
  cache_fs.write(config_string.c_str(), config_string.size());

  _odla_computation::instance()->session->compileAndExport(cache_fs.flush());
  cache_fs.flush();
  cache_fs.close();
  config_fs.close();
  POPLAR_CATCH
  return ret_value;
}

odla_status _odla_computation::init(bool is_compile) {
  if (!session) {
    popart::logging::warn("The computation:{} start to init", this);
    std::lock_guard<std::mutex> guard(init_mutex_);
    if (!session) {
      POPLAR_TRY
      odla_status status = set_opts();
      if (status != ODLA_SUCCESS) {
        popart::logging::err("set computation option failed");
        return status;
      }
      // Cretate the dataflow
      std::vector<popart::TensorId> ids;
      for (const auto& output : outputs_map)
        ids.push_back(output.second->tensor_id);

      popart::DataFlow data_flow(opts.batches_per_step, ids,
                                 popart::AnchorReturnType("All"));
      // Acquire IPU
      if (opts.use_ipu_model) {
        popart::logging::info("Using IPU Model to run.");
        std::map<std::string, std::string> deviceOpts{
            {"numIPUs", std::to_string(opts.ipu_num)}, {"tilesPerIPU", "1216"}};
        device =
            popart::DeviceManager::createDeviceManager().createIpuModelDevice(
                deviceOpts);
      } else
        device =
            popart::DeviceManager::createDeviceManager().acquireAvailableDevice(
                opts.ipu_num);

      if (nullptr == device) {
        popart::logging::err(
            "Failed to get a device when initializing odla_computation");
        throw std::runtime_error(
            "Failed to get a device when initializing odla_computation");
      }

      // Create and config SessionOptions
      set_session_opts();
      if (use_pipeline()) {
        builder = popart::Builder::createFromOnnxModel(set_pipeline_stage());
      }
      auto proto = builder->getModelProto(); // So, the init must be called at
                                             // odla_ExecuteCompute

      if (PopartConfig::instance()->load_onnx()) {
        popart::logging::info("Load onnx file as pipeline mode to run: {}",
                              PopartConfig::instance()->load_onnx_path());
        proto = PopartConfig::instance()->load_onnx_path();
      }

      if (PopartConfig::instance()->save_model()) {
        builder->saveModelProto(PopartConfig::instance()->save_model_path());
        popart::logging::info("The model saved to {}",
                              PopartConfig::instance()->save_model_path());
      }

      std::unique_ptr<popart::InferenceSession> new_session;
      // Create InferenceSession
      new_session = std::move(popart::InferenceSession::createFromOnnxModel(
          proto, data_flow, device, popart::InputShapeInfo(), session_opts_));

      if (!is_compile) {
        if (PopartConfig::instance()->load_or_save_cache()) {
          popart::logging::info("Load cachefile from existing stream");
          std::string version_string(popart::core::packageHash());
          if (!PopartConfig::instance()->sdk_version_match(version_string)) {
            popart::logging::err(
                "The sdk version of cache does not match popart hash {}",
                version_string);
            auto injector =
                PopartConfig::instance()->temp_get_error_inject_env();
            if (injector.empty())
              return ODLA_FAILURE;
            else
              popart::logging::warn("Error injector set, will compile");
          } else {
            auto cache_fs = PopartConfig::instance()->get_cache_fs();
            if (cache_fs->is_open()) {
              try {
                cache_fs->seekg(0, std::ios::beg);
                int config_len = 0;
                cache_fs->read((char*)&config_len, sizeof(config_len));
                cache_fs->seekg(config_len + sizeof(config_len), std::ios::beg);
                new_session->loadExecutableFromStream(*(cache_fs.get()));
              } catch (std::exception& e) {
                popart::logging::err("bad cache file: {}", e.what());
                return ODLA_FAILURE;
              } catch (...) {
                popart::logging::err("bad cache file");
                return ODLA_FAILURE;
              }
            }
          }
        }

        new_session->prepareDevice();
        new_session->setRandomSeed(0);  // Init seed
        new_session->weightsFromHost(); // Copy weights from host to IPU
      } else {
        is_compile_only_ = true;
      }
      // set session after all initialization done.
      session = std::move(new_session);
      // Thread must be started after all initialization done
      if (!is_compile) {
        ExecutionMode mode = PopartConfig::instance()->execution_mode();
        if (PIPELINE == mode || PARALLEL == mode || PIPELINE_ASYNC == mode) {
          std::thread parallel_thread(compute_loop, this);
          popart::logging::warn(
              "The computation: {}, parallel loop has been started", this);
          parallel_thread.detach();
        }
      }
      POPLAR_CATCH
    }
    popart::logging::warn(
        "The computation:{} has been initialised with session:{}", this,
        session.get());
  }
  return ODLA_SUCCESS;
}

// Now we set this by config file, should set by the caller?
odla_status _odla_computation::set_opts() {
  if (PopartConfig::instance()->debug()) {
    opts.ipu_num = PopartConfig::instance()->ipu_num();
    opts.batches_per_step = PopartConfig::instance()->batches_per_step();
  } else if (use_pipeline()) { // Only check when use pipeline
    if (opts.ipu_num != PopartConfig::instance()->ipu_num()) {
      popart::logging::err(
          "number of ipus in pipeline configuration:" +
          std::to_string(PopartConfig::instance()->ipu_num()) +
          " must same with options: " + std::to_string(opts.ipu_num));
      return ODLA_FAILURE;
    }
    if (opts.batches_per_step != PopartConfig::instance()->batches_per_step()) {
      popart::logging::warn(
          "batches per step in pipeline configuration:" +
          std::to_string(PopartConfig::instance()->batches_per_step()) +
          " not same with options: " + std::to_string(opts.batches_per_step));
      opts.batches_per_step = PopartConfig::instance()->batches_per_step();
    }
  }
  return ODLA_SUCCESS;
}

odla_status _odla_computation::set_executor() {
  odla_status ret_value = ODLA_SUCCESS;
  ExecutionMode mode = PopartConfig::instance()->execution_mode();
  if (PIPELINE == mode || PARALLEL == mode || PIPELINE_ASYNC == mode) {
    popart::logging::info("set the executor as parallel");
    executor_ = new Parallel();
  } else if (SEQUENCE == mode) {
    popart::logging::info("set the executor as sequence");
    executor_ = new Sequence();
  } else {
    popart::logging::err(
        "unknown excution mode: {}, Should be one of pipeline, parallel or "
        "sequence",
        std::to_string(mode));
    ret_value = ODLA_FAILURE;
  }
  return ret_value;
}

void _odla_computation::set_session_opts() {
  // This should be passed in by config file or some where
  if (use_pipeline()) {
    session_opts_.enablePipelining = true;
    // session_opts_.autoRecomputation = popart::RecomputationType::Pipeline;
    session_opts_.virtualGraphMode = popart::VirtualGraphMode::Manual;
  } else {
    session_opts_.virtualGraphMode = popart::VirtualGraphMode::Auto;
  }
  const char* envEngineCachePath = getenv("ENGINE_CACHE_PATH");
  if (opts.enable_engine_cache || envEngineCachePath != nullptr) {
    session_opts_.enableEngineCaching = true;
    session_opts_.cachePath =
        opts.enable_engine_cache ? opts.cache_dir : envEngineCachePath;
  }
  session_opts_.matmulOptions["use128BitConvUnitLoad"] = "true";
  session_opts_.matmulOptions["enableMultiStageReduce"] = "false";
  session_opts_.matmulOptions["enableFastReduce"] = "true";
  session_opts_.enableFloatingPointChecks = false;
  session_opts_.enableStochasticRounding = false;
  session_opts_.enablePrefetchDatastreams = false; // true;
  session_opts_.enableOutlining = true;
  std::string partials_type = "half";
  session_opts_.partialsTypeMatMuls = partials_type;
  session_opts_.convolutionOptions["partialsType"] = partials_type;
  session_opts_.outlineThreshold = 10.0;
  session_opts_.instrumentWithHardwareCycleCounter = false;
  session_opts_.disableGradAccumulationTensorStreams = true;
}

bool _odla_computation::hold() {
  auto this_thread_id = std::this_thread::get_id();
  std::stringstream ss;
  ss << std::this_thread::get_id();
  if (thread_id_of_holder == std::thread::id()) {
    thread_id_of_holder = this_thread_id;
    popart::logging::info("The odla_computation {} was held by thread {}", this,
                          this_thread_id);
    return true;
  } else if (thread_id_of_holder == this_thread_id) {
    return true;
  } else {
    std::stringstream ss_holder;
    ss_holder << thread_id_of_holder;
    popart::logging::warn(
        "The odla_computation {} has been held by thread: {}"
        ", when thread {} try to hold it.",
        this, thread_id_of_holder, this_thread_id);
  }
  return false;
}

std::string _odla_computation::set_pipeline_stage() {
  popart::logging::info("Setting pipeline stage for the model");
  std::stringstream input(builder->getModelProto());
  ONNX_NAMESPACE::ModelProto model_proto;
  google::protobuf::io::IstreamInputStream input_stream(&input);
  google::protobuf::io::CodedInputStream coded_input_stream(&input_stream);
  coded_input_stream.SetTotalBytesLimit(std::numeric_limits<int>::max(), -1);
  model_proto.ParseFromCodedStream(&coded_input_stream);
  popart::logging::info("Loaded the model for pipeline setting");
  auto ptr_graph = model_proto.mutable_graph();
  ONNX_NAMESPACE::GraphProto& graph = *ptr_graph;
  for (unsigned node_i = 0; node_i < graph.node_size(); node_i++) {
    auto ptr_node = graph.mutable_node(node_i);
    ONNX_NAMESPACE::NodeProto& node = *ptr_node;
    if (!node.has_name()) throw std::runtime_error("node of onnx has no name");
    int64_t ipu_idx = -1;
    int64_t pipeline_stage = -1;
    PopartConfig::instance()->get_pipeline_setting(node.name(), ipu_idx,
                                                   pipeline_stage);
    popart::logging::info("Node {} will be put to ipu {} stage {}", node.name(),
                          ipu_idx, pipeline_stage);
    bool found_ipu_att = false;
    bool found_stage_att = false;
    for (unsigned att_i = 0; att_i < node.attribute_size(); att_i++) {
      auto ptr_att = node.mutable_attribute(att_i);
      ONNX_NAMESPACE::AttributeProto& att = *ptr_att;
      if (att.name() == popart::sVirtualGraphAttribute) {
        found_ipu_att = true;
        att.set_i(ipu_idx);
      } else if (att.name() == popart::sPipelineStageAttribute) {
        found_stage_att = true;
        att.set_i(pipeline_stage);
      }
    }
    if (!found_ipu_att) {
      auto new_att = node.add_attribute();
      new_att->set_name(popart::sVirtualGraphAttribute);
      new_att->set_type(onnx::AttributeProto::INT);
      new_att->set_i(ipu_idx);
    }
    if (!found_stage_att) {
      auto new_att = node.add_attribute();
      new_att->set_name(popart::sPipelineStageAttribute);
      new_att->set_type(onnx::AttributeProto::INT);
      new_att->set_i(pipeline_stage);
    }
  }
  std::string pipelined_model;
  model_proto.SerializeToString(&pipelined_model);
  popart::logging::info("Pipeline stage has been set");
  return pipelined_model;
}

bool _odla_computation::use_pipeline() {
  static bool global_ipu_number_set = false;
  if (PopartConfig::instance()->no_pipeline()) {
    if (!global_ipu_number_set) {
      popart::logging::info(
          "PIPELINE not used for this run, "
          "Set the global virtual group to ipu 0");
      builder->setAttribute(popart::sVirtualGraphAttribute, 0);
      global_ipu_number_set = true;
    }
    return false;
  }
  return true;
}

bool _odla_context::hold(const std::string& function_name) {
  auto this_thread_id = std::this_thread::get_id();
  std::stringstream ss;
  ss << std::this_thread::get_id();
  if (thread_id_of_holder == std::thread::id()) // held by nobody
  {
    thread_id_of_holder = this_thread_id;
    popart::logging::info("[{}] The context {} has been held", this_thread_id,
                          this);
    return true;
  } else if (thread_id_of_holder == this_thread_id) { // held by this thread
    return true;
  } else { // held by other thread
    std::stringstream ss_holder;
    ss_holder << thread_id_of_holder;
    popart::logging::err(
        "[{}] odla_context {} has been held by thread: {}"
        ", when try to hold it in function {}. multi threads try to hold the "
        "same context.",
        this_thread_id, this, thread_id_of_holder, function_name);
    return false;
    //    throw std::runtime_error("Multiple threads try to hold the same
    //    context");
  }
  return false;
}

odla_status Sequence::compute(odla_computation comp, odla_context context,
                              odla_compute_mode mode, odla_device device) {
  std::lock_guard<std::mutex> comp_guard(sequence_mutex);
  popart::logging::info(">>> Sequence::compute() with ctx: {}", context);
  // Config StepIO
  std::map<popart::TensorId, popart::IArray&> inputs;
  for (auto& input : context->inputs) {
    inputs.emplace(input.first, *input.second);
  }
  std::map<popart::TensorId, popart::IArray&> outputs;
  for (auto& output : context->outputs) {
    outputs.emplace(output.first, *output.second);
  }
  static int i = 0;
  if (i == INT_MAX) i = 0;
  auto start = std::chrono::steady_clock::now();
  popart::StepIO stepio(inputs, outputs);
  // Run on ipu
  POPLAR_TRY
  comp->session->run(stepio);
  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  popart::logging::info("[ {} ] [Sequence::compute] takes {} s.", i++,
                        elapsed_seconds.count());
  popart::logging::info("<<< Sequence::compute() with ctx: {}", context);
  POPLAR_CATCH
  return ODLA_SUCCESS;
}

odla_status Parallel::compute(odla_computation comp, odla_context context,
                              odla_compute_mode mode, odla_device device) {
  popart::logging::info(">>> Parallel::compute() with context: {}", context);
  // Check whether the QueueManager status
  if (ODLA_SUCCESS == QManager::instance()->get_status()) {
    QManager::instance()->getQ()->put(
        context); // put the queues to wait list firstly
    context->wait();
  } else {
    popart::logging::err("Will return with status: {}",
                         QManager::instance()->get_status());
  }
  popart::logging::info("<<< Parallel::compute() with context {}", context);
  return QManager::instance()->get_status();
}
