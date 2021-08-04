//===- popart_config.cc ----------------------------------------------------===//
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

PopartConfig* PopartConfig::m_instance = new PopartConfig();

void PopartConfig::load_config(const std::string& file_path)
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
    // only set the split point, after which(included) the pipeline stage will be changed. So use the full name
    // If you don't know the exactly node, the regex is OK, at many setting should set the same value multiple times
    //set_pipeline_setting("layer12_", 1, 1);
}

bool PopartConfig::get_pipeline_setting(const std::string& node_name, int64_t &ipu_idx, int64_t& pipeline_stage)
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