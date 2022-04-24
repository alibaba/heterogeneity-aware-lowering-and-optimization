//===- popart_config.h ----------------------------------------------------===//
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

#ifndef POPART_CONFIG_H_
#define POPART_CONFIG_H_

#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <popart/version.hpp>
#include <regex>
#include <string>
#include <vector>

#include "ODLA/odla_common.h"
#include "json.hpp"
/**
 * The configuration format like follows:
 *
 * {
 *   "version":"1.0.0",
 *   "batch_per_step":10,
 *   "execution_mode":"pipeline",
 *   "ipu_num":2,
 *   "load_onnx":false,
 *   "load_onnx_path":"path",
 *   "pipeline":{
 *       "^embedding_"   : [0, 0],
 *       "^layer[0-9]_"  : [0, 0],
 *       "^layer1[0-1]_" : [0, 0],
 *       "^layer1[2-9]_" : [1, 1],
 *       "^layer2[0-3]_" : [1, 1],
 *       "^squad_"       : [1, 1]
 *   },
 *   "save_model" : true,
 *   "save_model_path":"pipeline_test.onnx"
 * }
 */
// When change ExecutionMode, must check PopartConfig::mode in popart_config.cc
enum ExecutionMode { UNKNOWN, PIPELINE, PARALLEL, SEQUENCE, PIPELINE_ASYNC };
using json = nlohmann::json;

class PopartConfig {
 private:
  float amp_;
  std::string version_;                 // Version of the configuration file
  std::string sdk_version_;             // version of the sdk
  int batches_per_step_;                // Batch per step
  static std::vector<std::string> mode; // string value of execution mode
  ExecutionMode execution_mode_; // The execution mode {PIPELINE, PARALLEL,
                                 // SEQUENCE, PIPELINE_ASYNC}
  bool load_onnx_; // Whether load onnx model to run instead of the model
                   // constructed. Use for test
  bool load_or_save_cache_; // If the session will load graph from cache
  std::string cache_path_;  // the path of cache file, for load cache
                            // directly

  std::string load_onnx_path_; // The path of onnx model file to load if
                               // load_onnx set to be true
  std::map<std::string, std::vector<int>>
      pipeline_setting_; // The pipeline settings if execution_mode was set as
                         // PIPELINE or PIPELINE_ASYNC
  bool save_model_;      // Whether save the mode constructed by model.cc
  std::string save_model_path_; // The path where to save the model if
                                // save_model was set as true
  int ipu_num_;                 // The number of ipu to use
  std::string queue_type_;      // the type of the queue used by parallel mode
  int queue_capacity_;          // the capacity of the queue
  bool debug_; // In debug mode, override the option setting by configuration
               // file
  std::string default_config_string_;

  bool inited_;

  std::shared_ptr<std::fstream> cache_fs;

  std::mutex config_mutex_;
  static PopartConfig* instance_;
  odla_status load_from_file(const std::string& file_path);

 public:
  PopartConfig()
      : version_("1.0.0"),
        sdk_version_("NA"),
        batches_per_step_(1),
        execution_mode_(UNKNOWN),
        load_onnx_(false),
        load_or_save_cache_(false),
        save_model_(false),
        inited_(false),
        ipu_num_(1) {}
  ~PopartConfig() {}

  void use_default();
  static PopartConfig* instance() { return instance_; }
  const std::string& version() { return version_; }
  inline void reset_init_state() {
    if (inited_) {
      std::lock_guard<std::mutex> guard(config_mutex_);
      if (inited_) {
        inited_ = false;
        if (cache_fs->is_open()) {
          cache_fs->close();
          cache_fs->clear();
        }
        pipeline_setting_.clear();
        sdk_version_ = "NA";
      }
    }
  }
  inline float amp() { return amp_; };
  inline int batches_per_step() { return batches_per_step_; }
  inline ExecutionMode execution_mode() { return execution_mode_; }
  inline bool load_onnx() { return load_onnx_; }
  inline bool save_model() { return save_model_; }
  inline const std::string& load_onnx_path() { return load_onnx_path_; }
  inline const std::string& save_model_path() { return save_model_path_; }
  inline const std::string& get_default_config_string() {
    return default_config_string_;
    ;
  }
  inline const int ipu_num() { return ipu_num_; }
  inline bool no_pipeline() { return pipeline_setting_.empty(); }
  inline std::string queue_type() { return queue_type_; }
  inline int queue_capacity() { return queue_capacity_; }
  inline bool debug() { return debug_; }
  inline bool inited() { return inited_; }
  inline std::shared_ptr<std::fstream> get_cache_fs() { return cache_fs; }
  inline void set_cache_fs(std::shared_ptr<std::fstream> fs) { cache_fs = fs; }

  inline bool load_or_save_cache() { return load_or_save_cache_; }
  inline const std::string& get_cache_path() { return cache_path_; }
  inline void set_load_or_save_cache(bool is_load_or_save_cache) {
    load_or_save_cache_ = is_load_or_save_cache;
  }
  inline void set_cache_path(const std::string& catch_path) {
    cache_path_ = catch_path;
  }

  bool sdk_version_match(std::string& sdk_version);
  void parse_from_json(const json&);
  odla_status load_from_string(const std::string& config_string);
  odla_status load_config(const char* file_path);
  bool get_pipeline_setting(const std::string& node_name, int64_t& ipu_idx,
                            int64_t& pipeline_stage);
  odla_status extract_config_from_cache();
  std::string temp_get_error_inject_env(
      const std::string& temp_config_path = "/tmp/temp_error_injector.json");

 private:
  void set_pipeline_setting(const std::string& name_pattern, int ipu_idx,
                            int pipeline_stage);
  void print();
};
#endif // POPART_CONFIG_H_
