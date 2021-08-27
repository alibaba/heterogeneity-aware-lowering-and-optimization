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
#include <mutex>
#include <iostream>
#include <chrono>
#include <popart/builder.hpp>
#include <popart/dataflow.hpp>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <bits/stdc++.h>
#include "onnx/onnx.pb.h"
#include "odla_popart.h"
#include "popart_config.h"
#include "odla_pipeline.h"

_odla_computation* _odla_computation::m_instance = new _odla_computation();

void compute_loop(odla_computation comp)
{
  //setup the stepio with allbacks
  popart::StepIOCallback stepio(input_callback,
                              input_complete_callback,
                              output_callback,
                              output_complete_callback);
  int i=0;
  while(!comp->is_done()){
    auto start = std::chrono::steady_clock::now();
    popart::logging::info("This is the {} time for the inference", i++);
    if(i == INT_MAX)
      i = 0;
    comp->session->run(stepio);
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    popart::logging::warn("[ {} ] ONE_STEP takes {} s. Check whether more inference tasks wating.", i, elapsed_seconds.count());
    //Make wait on CPU if there's not inference task
    start = std::chrono::steady_clock::now();
    while(!comp->is_done() && QManager::instance()->getQ()->size() == 0)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    end = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> elapsed_ms = end-start;
    popart::logging::warn("Found new tasks in {} ms.", elapsed_ms.count());
  }
  popart::logging::warn("The pipeline loop finished");
  comp->thread_complete_ = true;
}

void _odla_computation::init()
{
    if(!session){
        std::lock_guard<std::mutex> guard(m_init_mutex);
        if(!session){
            set_opts(); //Test code
            //Cretate the dataflow
            std::vector<popart::TensorId> ids;
            for (const auto& output : outputs_map)
                ids.push_back(output.second->tensor_id);
            popart::DataFlow data_flow(opts.batches_per_step, ids,
                                    popart::AnchorReturnType("All"));
            // Acquire IPU
            if(opts.use_ipu_model){
                popart::logging::info("Using IPU Model to run.");
                std::map<std::string, std::string> deviceOpts{
                    {"numIPUs", std::to_string(opts.ipu_num)}, {"tilesPerIPU", "1216"}};
                device = popart::DeviceManager::createDeviceManager().createIpuModelDevice(deviceOpts);
            }
            else
                device = popart::DeviceManager::createDeviceManager().acquireAvailableDevice(opts.ipu_num);
            // Create and config SessionOptions
            set_session_opts();

            builder = popart::Builder::createFromOnnxModel(set_pipeline_stage());
            auto proto = builder->getModelProto(); //So, the init must be called at odla_ExecuteCompute
            if(PopartConfig::instance()->load_onnx()){
                popart::logging::info("Load onnx file as pipeline mode to run: {}", 
                    PopartConfig::instance()->load_onnx_path());
                proto = PopartConfig::instance()->load_onnx_path();
            }
            if(PopartConfig::instance()->save_model()){
                builder->saveModelProto(PopartConfig::instance()->save_model_path());
                popart::logging::info("The model saved to {}", 
                    PopartConfig::instance()->save_model_path());
            }
            
            // Create InferenceSession
            auto new_session = popart::InferenceSession::createFromOnnxModel(
                proto,
                data_flow, 
                device, 
                popart::InputShapeInfo(), 
                m_session_opts
            );
            new_session->prepareDevice();
            new_session->setRandomSeed(0);  // Init seed
            new_session->weightsFromHost(); // Copy weights from host to IPU
            //If in parallel mode, start the thread
            ExecutionMode mode = PopartConfig::instance()->execution_mode();
            if(PIPELINE == mode || PARALLEL == mode){
                std::thread parallel_thread(compute_loop, this);
                popart::logging::warn("Parallel loop has been started");
                parallel_thread.detach();
            }
            session = std::move(new_session); //set session after all initialization done.
        }
    }
}

// Now we set this by config file, should set by the caller?
void _odla_computation::set_opts()
{
    if(PopartConfig::instance()->debug()) {
        opts.ipu_num = PopartConfig::instance()->ipu_num();
        opts.batches_per_step = PopartConfig::instance()->batches_per_step();
    } else {
        if(opts.ipu_num != PopartConfig::instance()->ipu_num())
            throw std::invalid_argument("number of ipus in pipeline configuration:"
              + std::to_string(PopartConfig::instance()->ipu_num()) 
              + " must same with options: " + std::to_string(opts.ipu_num));
        if(opts.batches_per_step != PopartConfig::instance()->batches_per_step())
            throw std::invalid_argument("batches per step in pipeline configuration:"
              + std::to_string(PopartConfig::instance()->batches_per_step()) 
              + " must same with options: " + std::to_string(opts.batches_per_step));
    }
}

void _odla_computation::set_executor()
{
    ExecutionMode mode = PopartConfig::instance()->execution_mode();
    if(PIPELINE == mode || PARALLEL == mode){
        popart::logging::info("set the executor as parallel");
        m_executor = new Parallel();
    }
    else if(SEQUENCE == mode){
        popart::logging::info("set the executor as sequence");
        m_executor = new Sequence();
    }
    else{
        throw std::invalid_argument(
            "*** FATAL *** unknown execution mode: {}" + std::to_string(mode)
            + ". Should be one of pipeline, parallel or sequence");
    }
}

void _odla_computation::set_session_opts()
{
    //This should be passed in by config file or some where
    if(!PopartConfig::instance()->no_pipeline()){
        m_session_opts.enablePipelining = true;
        //m_session_opts.autoRecomputation = popart::RecomputationType::Pipeline;
        m_session_opts.virtualGraphMode = popart::VirtualGraphMode::Manual;
    }else{
        m_session_opts.virtualGraphMode = popart::VirtualGraphMode::Auto;
    }
    //m_session_opts.matmulOptions["use128BitConvUnitLoad"] = "true";
    //m_session_opts.matmulOptions["enableMultiStageReduce"] = "false";
    //m_session_opts.matmulOptions["enableFastReduce"] = "true";
    m_session_opts.enableFloatingPointChecks = false;
    m_session_opts.enableStochasticRounding = false;
    m_session_opts.enablePrefetchDatastreams = false; //true;
    m_session_opts.enableOutlining = true;
    std::string partials_type = "half";
    m_session_opts.partialsTypeMatMuls = partials_type;
    m_session_opts.convolutionOptions["partialsType"] = partials_type;
    m_session_opts.outlineThreshold = 10.0;
    m_session_opts.instrumentWithHardwareCycleCounter = false;
    m_session_opts.disableGradAccumulationTensorStreams = true;
}

bool _odla_computation::hold()
{
    auto this_thread_id = std::this_thread::get_id();
    std::stringstream ss;
    ss << std::this_thread::get_id();
    if(thread_id_of_holder == std::thread::id())
    {
        thread_id_of_holder = this_thread_id;
        popart::logging::info("The odla_computation {} was held by thread {}", this, this_thread_id);
        return true;
    }else if(thread_id_of_holder == this_thread_id){
        return true;
    }else{
        std::stringstream ss_holder;
        ss_holder << thread_id_of_holder;
        //throw std::runtime_error("The odla_computation has been held by thread: " 
        //      + ss_holder.str() + ", when thread" + ss.str() + " try to hold it.");
        popart::logging::err("The odla_computation {} has been held by thread: {}"
              ", when thread {} try to hold it.", this, thread_id_of_holder, this_thread_id);
    }
    return false;
}

std::string _odla_computation::set_pipeline_stage() {
    popart::logging::info("Setting pipeline stage for the model");
    std::stringstream input(this->builder->getModelProto());
    ONNX_NAMESPACE::ModelProto model_proto;
    google::protobuf::io::IstreamInputStream input_stream(&input);
    google::protobuf::io::CodedInputStream coded_input_stream(&input_stream);
    coded_input_stream.SetTotalBytesLimit(std::numeric_limits<int>::max(), -1);
    model_proto.ParseFromCodedStream(&coded_input_stream);
    popart::logging::info("Loaded the model for pipeline setting");
    auto ptr_graph = model_proto.mutable_graph();
    ONNX_NAMESPACE::GraphProto &graph = *ptr_graph;
    for(unsigned node_i = 0; node_i < graph.node_size(); node_i++) {
        auto ptr_node = graph.mutable_node(node_i);
        ONNX_NAMESPACE::NodeProto &node = *ptr_node;
        if(!node.has_name())
            throw std::runtime_error("node of onnx has no name");
        int64_t ipu_idx = -1;
        int64_t pipeline_stage = -1;
        PopartConfig::instance()->get_pipeline_setting(node.name(), ipu_idx, pipeline_stage);
        popart::logging::info("Node {} will be put to ipu {} stage {}", node.name(), ipu_idx, pipeline_stage);
        bool found_ipu_att = false;
        bool found_stage_att = false;
        for(unsigned att_i = 0; att_i < node.attribute_size(); att_i++) {
            auto ptr_att = node.mutable_attribute(att_i);
            ONNX_NAMESPACE::AttributeProto &att = *ptr_att;
            if(att.name() == popart::sVirtualGraphAttribute){
                found_ipu_att = true;
                att.set_i(ipu_idx);
            }
            else if(att.name() == popart::sPipelineStageAttribute){
                found_stage_att = true;
                att.set_i(pipeline_stage);
            }
        }
        if(!found_ipu_att){
            auto new_att = node.add_attribute();
            new_att->set_name(popart::sVirtualGraphAttribute);
            new_att->set_type(onnx::AttributeProto::INT);
            new_att->set_i(ipu_idx);
        }
        if(!found_stage_att){
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

bool _odla_computation::use_pipeline()
{
    static bool global_ipu_number_set = false;
    if(PopartConfig::instance()->no_pipeline()){
        if(!global_ipu_number_set){
            popart::logging::info("PIPELINE not used for this run, "
                "Set the global virtual group to ipu 0");
            builder->setAttribute(popart::sVirtualGraphAttribute, 0);
            global_ipu_number_set = true;
        }
        return false;
    }
    return true;
}

bool _odla_context::hold(const std::string& function_name)
{
    auto this_thread_id = std::this_thread::get_id();
    std::stringstream ss;
    ss << std::this_thread::get_id();
    if(thread_id_of_holder == std::thread::id()) // held by nobody
    {
        thread_id_of_holder = this_thread_id;
        popart::logging::info("[{}] The context {} has been held", this_thread_id, this);
        return true;
    }else if(thread_id_of_holder == this_thread_id){ // held by this thread
        return true;
    }else{ //held by other thread
        std::stringstream ss_holder;
        ss_holder << thread_id_of_holder;
        popart::logging::err("[{}] odla_context {} has been held by thread: {}" 
              ", when try to hold it in function {}.", 
              this_thread_id, this, thread_id_of_holder, function_name);
        throw std::runtime_error("Multiple threads try to hold the same context");
    }
    return false;
}

void Sequence::compute(odla_computation comp, odla_context context,
                                odla_compute_mode mode, odla_device device) 
{
    //comp->init();
    std::lock_guard<std::mutex> comp_guard(sequence_mutex);
    popart::logging::info( ">>> Sequence::compute() with ctx: {}", context);
    // Config StepIO
    std::map<popart::TensorId, popart::IArray&> inputs;
    for (auto& input : context->inputs) {
        inputs.emplace(input.first, *input.second);
    }
    std::map<popart::TensorId, popart::IArray&> outputs;
    for (auto& output : context->outputs) {
        outputs.emplace(output.first, *output.second);
    }
    static int i=0;
    if(i == INT_MAX)
      i = 0;
    auto start = std::chrono::steady_clock::now();
    popart::StepIO stepio(inputs, outputs);
    // Run on ipu
    comp->session->run(stepio);
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    popart::logging::info("[ {} ] [Sequence::compute] takes {} s.", 
        i++, elapsed_seconds.count());
    popart::logging::info("<<< Sequence::compute() with ctx: {}", context);
}

void Parallel::compute(odla_computation comp, odla_context context,
                       odla_compute_mode mode,odla_device device) 
{
    popart::logging::info(">>> Parallel::compute() with context: {}", context);
    QManager::instance()->getQ()->put(context); //put the queues to wait list firstly
    //comp->init();
    context->wait();
    popart::logging::info("<<< Parallel::compute() with context {}", context);
}