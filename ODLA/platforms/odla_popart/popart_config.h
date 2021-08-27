//===- odla_popart.h ------------------------------------------------------===//
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

#include <string>
#include <map>
#include <vector>
#include <regex>
#include <iostream>
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
enum ExecutionMode {UNKNOWN, PIPELINE, PARALLEL, SEQUENCE};

class PopartConfig{
private:
    float amp_;
    std::string version_;    // Version of the configuration file
    int batches_per_step_;     // Batch per step for PIPELINE & PARALLEL execution mode
    ExecutionMode execution_mode_;   // The execution mode {PIPELINE, PARALLEL, SEQUENCE}
    bool load_onnx_; // Whether load onnx model to run instead of the model constructed. Use for test
    std::string load_onnx_path_; // The path of onnx model file to load if load_onnx set to be true
    std::map<std::string, std::vector<int>> pipeline_setting_;   // The pipeline settings if execution_mode was set as PIPELINE
    bool save_model_;    // Whether save the mode constructed by model.cc
    std::string save_model_path_;    // The path where to save the model if save_model was set as true
    int ipu_num_;    // The number of ipu to use
    std::string queue_type_;     //the type of the queue used by parallel mode
    int queue_capacity_;         //the capacity of the queue
    bool debug_;    // In debug mode, override the option setting by configuration file
    static PopartConfig* instance_;
    void use_default();
    void load_from_file(const std::string& file_path);
public:
    PopartConfig(): version_("1.0.0"), 
                    batches_per_step_(1), execution_mode_(UNKNOWN),
                    load_onnx_(false), save_model_(false), ipu_num_(1)
                    {/*std::cout << "PopartConfig instance created" << std::endl;*/}
    ~PopartConfig(){}
    static PopartConfig* instance(){return instance_;}
    const std::string& version(){return version_;}
    inline float amp(){return amp_;};
    inline int batches_per_step(){return batches_per_step_;}
    inline ExecutionMode execution_mode(){return execution_mode_;}
    inline bool load_onnx(){return load_onnx_;}
    inline bool save_model(){return save_model_;}
    inline const std::string& load_onnx_path(){return load_onnx_path_;}
    inline const std::string& save_model_path(){return save_model_path_;}
    inline const int ipu_num(){return ipu_num_;}
    inline bool no_pipeline(){return pipeline_setting_.empty();}
    inline std::string queue_type(){return queue_type_;}
    inline int queue_capacity(){return queue_capacity_;}
    inline bool debug(){return debug_;}

    void load_config(const char* file_path);
    bool get_pipeline_setting(const std::string& node_name, int64_t &ipu_idx, int64_t& pipeline_stage);

private:
    void set_pipeline_setting(const std::string& name_pattern, int ipu_idx, int pipeline_stage);
    void print();
};
#endif//POPART_CONFIG_H_