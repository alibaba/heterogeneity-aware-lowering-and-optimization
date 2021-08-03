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
//#include "odla_pipeline.h"

/**
 * The attributes need for the configuration
 * 
 *  {
 *      version:"1.0.0"
 *      batch_per_step:number
 *      execution_mode:"pipeline" | "parallel" | "sequence",
 *      ipu_num:number
 *      load_onnx:false | true,
 *      load_onnx_path:"path",
 *      
 *      pipeline:{
 *          "regex_1" : [ipu_num, pipeline_stage],
 *          "regex_2" : [ipu_num, pipeline_stage],
 *          ...
 *          "regex_3" : [ipu_num, pipeline_stage]
 *      },
 *      save_model:false | true,
 *      save_model_path:"path"
 *  }
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
                    m_load_onnx(false), m_save_model(false), m_ipu_num(1){}
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

    void load_config(const std::string& file_path)
    {
        //We need to implement the file reader  here
        m_batch_per_step = 10;
        m_execution_mode = PIPELINE;
        m_load_onnx = false;
        m_save_model = true;
        m_load_onnx_path = "/home/jackz/repos/heterogeneity-aware-lowering-and-optimization/MLPerf/pipeline/new_halo.onnx";
        m_save_model_path = "pipeline_test.onnx";
        m_ipu_num = 2;
        //set the pipeline setting
        set_pipeline_setting("^embedding_", 0, 0);
        for(int i = 0; i < 24; i++){
            int ipu_idx = 0;
            int pipeline_stage = 0;
            std::string pattern = "^layer" + std::to_string(i) + "_";
            if(i >= 12 ){
                ipu_idx = 1;
                pipeline_stage = 1;
            }
            set_pipeline_setting(pattern, ipu_idx, pipeline_stage);
        }
        set_pipeline_setting("^squad_", 1, 1);
    }

    bool get_pipeline_setting(const std::string& node_name, int64_t &ipu_idx, int64_t& pipeline_stage)
    {
        for(auto &v : m_pipeline_setting){
            auto name_pattern = std::regex(v.first, std::regex::icase);
            auto found = std::regex_search(node_name, name_pattern);
            if(found){
                std::cout << "node name: " << node_name << " matched with pattern: " << v.first 
                        << ", will be put in ipu: " << v.second[0] 
                        << ", pipeline stage: " << v.second[1] << std::endl;
                ipu_idx = v.second[0];
                pipeline_stage = v.second[1];
                return true;
            }
        }
        std::cerr << "*** Oops *** the node name: " << node_name << " did not match to any pattern !" << std::endl;
        return false;
    }

private:
    void set_pipeline_setting(const std::string& name_pattern, int ipu_idx, int pipeline_stage)
    {
        std::vector<int> values;
        values.push_back(ipu_idx);
        values.push_back(pipeline_stage);
        m_pipeline_setting[name_pattern] = values;
    }
};
#endif//POPART_CONFIG_H_