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
    std::string m_version;    // Version of the configuration file
    int m_batch_per_step;     // Batch per step for PIPELINE & PARALLEL execution mode
    ExecutionMode m_execution_mode;   // The execution mode {PIPELINE, PARALLEL, SEQUENCE}
    bool m_load_onnx; // Whether load onnx model to run instead of the model constructed. Use for test
    std::string m_load_onnx_path; // The path of onnx model file to load if load_onnx set to be true
    std::map<std::string, std::vector<int>> m_pipeline_setting;   // The pipeline settings if execution_mode was set as PIPELINE
    bool m_save_model;    // Whether save the mode constructed by model.cc
    std::string m_save_model_path;    // The path where to save the model if save_model was set as true
    int m_ipu_num;    // The number of ipu to use
    static PopartConfig* m_instance;
public:
    PopartConfig(): m_version("1.0.0"), 
                    m_batch_per_step(1), m_execution_mode(UNKNOWN),
                    m_load_onnx(false), m_save_model(false), m_ipu_num(1)
                    {std::cout << "PopartConfig instance created" << std::endl;}
    ~PopartConfig(){}
    static PopartConfig* instance(){return m_instance;}
    const std::string& version(){return m_version;}
    int batch_per_step(){return m_batch_per_step;}
    ExecutionMode execution_mode(){return m_execution_mode;}
    bool load_onnx(){return m_load_onnx;}
    bool save_model(){return m_save_model;}
    const std::string& load_onnx_path(){return m_load_onnx_path;}
    const std::string& save_model_path(){return m_save_model_path;}
    const int ipu_num(){return m_ipu_num;}
    bool no_pipeline(){return m_pipeline_setting.empty();}

    void load_config(const std::string& file_path);
    bool get_pipeline_setting(const std::string& node_name, int64_t &ipu_idx, int64_t& pipeline_stage);

private:
    void set_pipeline_setting(const std::string& name_pattern, int ipu_idx, int pipeline_stage);
    void print();
};
#endif//POPART_CONFIG_H_