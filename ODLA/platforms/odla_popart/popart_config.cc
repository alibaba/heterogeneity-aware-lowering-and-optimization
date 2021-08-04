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

#include <fstream>
#include <iostream>
#include "popart_config.h"
#include "json.hpp"
#include <typeinfo>

PopartConfig* PopartConfig::m_instance = new PopartConfig();

void PopartConfig::load_config(const std::string& file_path)
{
    using json = nlohmann::json;
    //std::ifstream ifs("/home/jackz/repos/heterogeneity-aware-lowering-and-optimization/ODLA/platforms/odla_popart/config.json");
    std::ifstream ifs(file_path);
    json jf = json::parse(ifs);

    m_version           = jf["version"].get<std::string>();
    m_batch_per_step    = jf["batch_per_step"].get<int>();
    m_ipu_num           = jf["ipu_num"].get<int>();
    m_save_model        = jf["save_model"].get<bool>();
    m_save_model_path   = jf["save_model_path"].get<std::string>();
    m_load_onnx         = jf["load_onnx"].get<bool>();
    m_load_onnx_path    = jf["load_onnx_path"].get<std::string>();
    std::string execution_mode = jf["execution_mode"].get<std::string>();
    if("pipeline" == execution_mode)
        m_execution_mode = PIPELINE;
    else if("parallel" == execution_mode)
        m_execution_mode = PARALLEL;
    else
        m_execution_mode = SEQUENCE;

    const json& rh = jf["pipeline"];
    for (auto& element : rh.items()) {
        set_pipeline_setting(element.key(), element.value()[0], element.value()[1]);
    }

    print();
    exit(0);
}

void PopartConfig::print()
{
    std::string line(80, '=');
    std::cout << line << std::endl;
    std::cout << "version: " << m_version << std::endl;
    std::cout << "batch_per_step: " << m_batch_per_step << std::endl;
    std::string mode[] = {"UNKNOWN", "PIPELINE", "PARALLEL", "SEQUENCE"};
    std::cout << "execution_mode: " << mode[(long unsigned int)m_execution_mode] << std::endl;
    std::cout << "ipu_num: " << m_ipu_num << std::endl;
    std::string bool_value[] = {"false", "true"};
    std::cout << "load_onnx: " << bool_value[(long unsigned int)m_load_onnx] << std::endl;
    std::cout << "load_onnx_path: " << m_load_onnx_path << std::endl << std::endl;
    std::cout << "save_model: " << bool_value[(long unsigned int)m_save_model] << std::endl;
    std::cout << "save_model_path: " << m_save_model_path << std::endl;
    for(auto &a : m_pipeline_setting)
        std::cout << a.first << " <-----> [" << a.second[0] << ", " << a.second[1] << "]" << std::endl;
    std::cout << line << std::endl;
}

void PopartConfig::set_pipeline_setting(const std::string& name_pattern, int ipu_idx, int pipeline_stage)
{
    std::vector<int> values;
    values.push_back(ipu_idx);
    values.push_back(pipeline_stage);
    m_pipeline_setting[name_pattern] = values;
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