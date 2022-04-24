//===- popart_config.cc
//----------------------------------------------------===//
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

#include "popart_config.h"

#include <fstream>
#include <iostream>
#include <popart/logging.hpp>
#include <typeinfo>

#include "json.hpp"

PopartConfig* PopartConfig::instance_ = new PopartConfig();
std::vector<std::string> PopartConfig::mode = {
    "unknown", "pipeline", "parallel", "sequence", "pipeline_async"};

const char* bool_to_str(const bool& value) { return value ? "true" : "false"; }

const std::string get_config_path_from_cache_file(
    const std::string& cache_path) {
  std::string file_suffix(".popart");
  int file_prefix = cache_path.rfind(file_suffix);
  if (file_prefix == std::string::npos ||
      file_prefix + file_suffix.size() < cache_path.size()) {
    popart::logging::err(
        "Bad cache file name. File name should end with '.popart'");
    return std::string("");
  }
  return std::string(cache_path.substr(0, file_prefix) + ".json");
}

void PopartConfig::use_default() {
  amp_ = 0.6;
  sdk_version_ = popart::core::packageHash();
  version_ = "1.0.0";
  batches_per_step_ = 1;
  ipu_num_ = 1;
  save_model_ = false;
  save_model_path_ = "odla_popart_saved.onnx";
  load_onnx_ = false;
  load_onnx_path_ = "not_set.onnx";
  execution_mode_ = SEQUENCE;
  queue_type_ = "LockFreeQueue";
  queue_capacity_ = 1024 * 1024;
  debug_ = false;
  char config_base[] =
      "{\n\
      \"sdk_version\":\"%s\",\n\
      \"version\":\"%s\",\n\
      \"amp\":%f,\n\
      \"batches_per_step\":%d,\n\
      \"execution_mode\":\"%s\",\n\
      \"ipu_num\":%d,\n\
      \"load_onnx\":%s, \n\
      \"load_onnx_path\":\"%s\",\n\
      \"queue_type\":\"%s\",\n\
      \"queue_capacity\":%d,\n\
      \"debug\":%s\n\
      }\n";
  char raw_default_config[1024] = {0};
  snprintf(raw_default_config, 1024, config_base, sdk_version_.c_str(),
           version_.c_str(), amp_, batches_per_step_,
           PopartConfig::mode[(int)execution_mode_].c_str(), ipu_num_,
           bool_to_str(load_onnx_), load_onnx_path_.c_str(),
           queue_type_.c_str(), queue_capacity_, bool_to_str(debug_));
  default_config_string_.assign(raw_default_config);
}

odla_status PopartConfig::load_config(const char* env_file_path) {
  if (inited_) {
    popart::logging::info("config already inited");
    return ODLA_SUCCESS;
  }
  odla_status ret = ODLA_FAILURE;
  if (load_or_save_cache()) {
    ret = extract_config_from_cache();
    if (ret != ODLA_SUCCESS) {
      popart::logging::warn("load config from cache failed");
      std::string config_file_path =
          get_config_path_from_cache_file(std::string(cache_path_));
      if (!config_file_path.empty()) {
        popart::logging::info("try load from file: {}", config_file_path);
        ret = load_from_file(config_file_path);
      }
    }
  }
  if (ret != ODLA_SUCCESS) {
    use_default();
    if (env_file_path != nullptr) {
      ret = load_from_file(env_file_path);
      if (ret != ODLA_SUCCESS) {
        popart::logging::info("use default config");
      }
    } else {
      popart::logging::info("use default config");
    }
  }
  return ODLA_SUCCESS;
}

void PopartConfig::parse_from_json(const json& jf) {
  if (jf.contains("sdk_version")) {
    sdk_version_ = jf["sdk_version"].get<std::string>();
  }
  if (jf.contains("amp")) {
    amp_ = jf["amp"].get<float>();
  }
  if (jf.contains("version")) {
    version_ = jf["version"].get<std::string>();
  }
  if (jf.contains("batches_per_step")) {
    batches_per_step_ = jf["batches_per_step"].get<int>();
  }
  if (jf.contains("ipu_num")) {
    ipu_num_ = jf["ipu_num"].get<int>();
  }
  if (jf.contains("save_model")) {
    save_model_ = jf["save_model"].get<bool>();
  }
  if (jf.contains("save_model_path")) {
    save_model_path_ = jf["save_model_path"].get<std::string>();
  }
  if (jf.contains("load_onnx")) {
    load_onnx_ = jf["load_onnx"].get<bool>();
  }
  if (jf.contains("load_onnx_path")) {
    load_onnx_path_ = jf["load_onnx_path"].get<std::string>();
  }
  if (jf.contains("execution_mode")) {
    std::string execution_mode = jf["execution_mode"].get<std::string>();
    if ("pipeline" == execution_mode)
      execution_mode_ = PIPELINE;
    else if ("parallel" == execution_mode)
      execution_mode_ = PARALLEL;
    else if ("pipeline_async" == execution_mode)
      execution_mode_ = PIPELINE_ASYNC;
    else
      execution_mode_ = SEQUENCE;
  }
  if (jf.contains("pipeline")) {
    const json& rh = jf["pipeline"];
    for (auto& element : rh.items()) {
      set_pipeline_setting(element.key(), element.value()[0],
                           element.value()[1]);
    }
  }
  if (jf.contains("queue_type"))
    queue_type_ = jf["queue_type"].get<std::string>();
  else
    queue_type_ = "LockFreeQueue";

  if (jf.contains("queue_capacity"))
    queue_capacity_ = jf["queue_capacity"].get<int>();
  else
    queue_capacity_ = 1024 * 1024;
  if (jf.contains("debug")) debug_ = jf["debug"].get<bool>();
  print();

  inited_ = true;
}

odla_status PopartConfig::load_from_string(const std::string& config_string) {
  if (inited_) {
    return ODLA_SUCCESS;
  }
  json jf;
  try {
    jf = json::parse(config_string);
  } catch (std::exception& e) {
    popart::logging::err("parse config falied:{}", e.what());
    return ODLA_FAILURE;
  } catch (...) {
    popart::logging::err("parse config falied");
    return ODLA_FAILURE;
  }
  parse_from_json(jf);
  return ODLA_SUCCESS;
}

std::string PopartConfig::temp_get_error_inject_env(
    const std::string& temp_config_path) {
  using json = nlohmann::json;
  std::ifstream ifs(temp_config_path, std::ios_base::in);
  popart::logging::warn("get injector info from config file: {}",
                        temp_config_path);
  if (!ifs.good()) {
    popart::logging::warn("config file {} not found, no injector for this run",
                          temp_config_path);
    return std::string("");
  }
  try {
    json jf = json::parse(ifs);
    if (jf.contains("POPLAR_ENGINE_OPTIONS")) {
      auto poplar_engine_options =
          jf["POPLAR_ENGINE_OPTIONS"].get<std::string>();
      popart::logging::info(
          "Read the POPLAR_ENGINE_OPTIONS from file:{} with value: {}.",
          temp_config_path, poplar_engine_options);
      return poplar_engine_options;
    }
  } catch (std::exception& e) {
    popart::logging::err("parse config falied:{}", e.what());
    return std::string("");
  } catch (...) {
    popart::logging::err("parse config falied");
    return std::string("");
  }
  return std::string("");
}

odla_status PopartConfig::load_from_file(const std::string& file_path) {
  if (inited_) {
    return ODLA_SUCCESS;
  }
  using json = nlohmann::json;
  std::ifstream ifs(file_path, std::ios_base::in);
  if (!ifs.good()) {
    popart::logging::err("config file {} not found", file_path);
    return ODLA_FAILURE;
  }
  json jf = json::parse(ifs);
  parse_from_json(jf);
  return ODLA_SUCCESS;
}

void PopartConfig::print() {
  std::string line(80, '=');
  popart::logging::info(line);
  popart::logging::info("sdk_version: {}", sdk_version_);
  popart::logging::info("version: {}", version_);
  popart::logging::info("amp: {}", amp_);
  popart::logging::info("batch_per_step: {}", batches_per_step_);
  popart::logging::info("execution_mode: {}",
                        PopartConfig::mode[(long unsigned int)execution_mode_]);
  popart::logging::info("ipu_num: {}", ipu_num_);
  std::string bool_value[] = {"false", "true"};
  popart::logging::info("load_onnx: {}",
                        bool_value[(long unsigned int)load_onnx_]);
  popart::logging::info("load_onnx_path: {}", load_onnx_path_);
  popart::logging::info("save_model: {}",
                        bool_value[(long unsigned int)save_model_]);
  popart::logging::info("save_model_path: {}", save_model_path_);
  popart::logging::info("queue_type: {}", queue_type_);
  popart::logging::info("queue_capacity: {}", queue_capacity_);
  popart::logging::info("debug: {}", bool_value[(long unsigned int)debug_]);
  popart::logging::info("pipeline configuration:");
  for (auto& a : pipeline_setting_)
    popart::logging::info("{} <-----> [{}, {}]", a.first, a.second[0],
                          a.second[1]);
  popart::logging::info(line);
}

void PopartConfig::set_pipeline_setting(const std::string& name_pattern,
                                        int ipu_idx, int pipeline_stage) {
  std::vector<int> values;
  values.push_back(ipu_idx);
  values.push_back(pipeline_stage);
  pipeline_setting_[name_pattern] = values;
}

bool PopartConfig::get_pipeline_setting(const std::string& node_name,
                                        int64_t& ipu_idx,
                                        int64_t& pipeline_stage) {
  for (auto& v : pipeline_setting_) {
    auto name_pattern = std::regex(v.first, std::regex::icase);
    auto found = std::regex_search(node_name, name_pattern);
    if (found) {
      popart::logging::debug(
          "node name: {} matched with pattern: {}"
          ", will be put in ipu: {}, pipeline stage: {}",
          node_name, v.first, v.second[0], v.second[1]);
      ipu_idx = v.second[0];
      pipeline_stage = v.second[1];
      return true;
    }
  }
  auto default_setting_iter = pipeline_setting_.find("^__all_unmatched__$");
  if (default_setting_iter != pipeline_setting_.end()) {
    ipu_idx = default_setting_iter->second[0];
    pipeline_stage = default_setting_iter->second[1];
    return true;
  }

  throw std::runtime_error(
      "Node: " + node_name +
      " was not configured to any ipu or stage for pipeline");
  return false;
}

odla_status PopartConfig::extract_config_from_cache() {
  cache_fs = std::make_shared<std::fstream>(
      cache_path_,
      std::ios_base::in | std::ios_base::out | std::ifstream::binary);
  int config_len = 0;
  popart::logging::info("load config from cache file: {}", cache_path_.c_str());
  if (!cache_fs->is_open()) {
    popart::logging::err("cache file is not exist");
    return ODLA_FAILURE;
  }
  if (cache_fs->read((char*)&config_len, sizeof(config_len))) {
    std::vector<char> config_data_buffer(config_len);
    if (cache_fs->read(config_data_buffer.data(), config_len)) {
      std::string config_string(config_data_buffer.begin(),
                                config_data_buffer.end());

      odla_status ret = load_from_string(config_string);
      if (ret != ODLA_SUCCESS) {
        popart::logging::err("load from cached config string failed.");
        return ODLA_FAILURE;
      }
    }
  }
  return ODLA_SUCCESS;
}

bool PopartConfig::sdk_version_match(std::string& sdk_version) {
  popart::logging::warn(
      "sdk version in the cache file is {}, and from sdk is {}", sdk_version_,
      sdk_version);
  return (sdk_version_.compare(sdk_version) == 0);
}
