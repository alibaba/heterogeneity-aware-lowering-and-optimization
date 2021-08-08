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
#include <popart/logging.hpp>

PopartConfig* PopartConfig::m_instance = new PopartConfig();

void PopartConfig::load_config(const std::string& file_path)
{
    using json = nlohmann::json;
    std::ifstream ifs(file_path);
    json jf = json::parse(ifs);

    amp_                = jf["amp"].get<float>();
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

    if(jf.contains("queue_type"))
        queue_type_ = jf["queue_type"].get<std::string>();
    else
        queue_type_ = "LockFreeQueue";
    
    if(jf.contains("queue_capacity"))
        queue_capacity_ = jf["queue_capacity"].get<int>();
    else
        queue_capacity_ = 1024;
    print();
}

void PopartConfig::print()
{
    std::string line(80, '=');
    popart::logging::info(line);
    popart::logging::info("version: {}", m_version);
    popart::logging::info("amp: {}", amp_);
    popart::logging::info("batch_per_step: {}", m_batch_per_step);
    std::string mode[] = {"UNKNOWN", "PIPELINE", "PARALLEL", "SEQUENCE"};
    popart::logging::info("execution_mode: {}",
        mode[(long unsigned int)m_execution_mode]);
    popart::logging::info("ipu_num: {}", m_ipu_num);
    std::string bool_value[] = {"false", "true"};
    popart::logging::info("load_onnx: {}", 
        bool_value[(long unsigned int)m_load_onnx]);
    popart::logging::info("load_onnx_path: {}", m_load_onnx_path);
    popart::logging::info("save_model: {}", 
        bool_value[(long unsigned int)m_save_model]);
    popart::logging::info("save_model_path: {}", m_save_model_path);
    popart::logging::info("queue_type: {}", queue_type_);
    popart::logging::info("queue_capacity: {}", queue_capacity_);
    popart::logging::info("pipeline configuration:");
    for(auto &a : m_pipeline_setting)
        popart::logging::info("{} <-----> [{}, {}]",
            a.first, a.second[0], a.second[1]);
    popart::logging::info(line);
}

void PopartConfig::set_pipeline_setting(
    const std::string& name_pattern, int ipu_idx, int pipeline_stage)
{
    std::vector<int> values;
    values.push_back(ipu_idx);
    values.push_back(pipeline_stage);
    m_pipeline_setting[name_pattern] = values;
}

bool PopartConfig::get_pipeline_setting(
    const std::string& node_name, int64_t &ipu_idx, int64_t& pipeline_stage)
{
    for(auto &v : m_pipeline_setting){
        auto name_pattern = std::regex(v.first, std::regex::icase);
        auto found = std::regex_search(node_name, name_pattern);
        if(found){
            popart::logging::info("node name: {} matched with pattern: {}"
                ", will be put in ipu: {}, pipeline stage: {}", 
                node_name, v.first, v.second[0], v.second[1]);
            ipu_idx = v.second[0];
            pipeline_stage = v.second[1];
            return true;
        }
    }
    popart::logging::info(
        "*** Oops *** the node name: {} did not match to any pattern!", 
        node_name);
    return false;
}