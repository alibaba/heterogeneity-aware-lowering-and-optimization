//===- odla_pipeline.cc ----------------------------------------------------===//
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

#include <ODLA/odla.h>
#include <dlfcn.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <functional>
#include <memory>
#include <numeric>
#include <popart/builder.hpp>
#include <popart/dataflow.hpp>
#include <popart/devicemanager.hpp>
#include <popart/names.hpp>
#include <popart/ndarraywrapper.hpp>
#include <popart/session.hpp>
#include <popart/sessionoptions.hpp>
#include <popart/stepio.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/voiddata.hpp>
#include <popart/names.hpp>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <mutex>
#include <queue>
#include <thread>
#include <array>
#include <fstream>
#include <sstream>
#include <bits/stdc++.h>

#include "ODLA/odla_common.h"
#include "common.h"
#include "odla_popart.h"
#include "odla_pipeline.h"

#if !defined(ODLA_VERSION_NUMBER) || (ODLA_VERSION_NUMBER < 50)
#error This library requires minimum ODLA version 0.5
#endif

ContextQueues* ContextQueues::p_context_queues = nullptr;
std::mutex ContextQueues::instance_mutex;

static odla_context shared_data = nullptr;
static std::mutex shared_data_mutext;

odla_context create_empty_odla_context()
{
  if(!shared_data){
    std::lock_guard<std::mutex> guard(shared_data_mutext);
    if(!shared_data){
      std::cout << "Creating the shared data for the _odla_pipeline_zero" << std::endl;
      shared_data = new _odla_context(SingleComp::get_instance()->get_comp());
      for(auto& value : shared_data->comp->input_values){
        std::size_t num_elements = 1;
        for(auto& shape : value->tensor_info.shape())
          num_elements *= shape;
        float* data = new float[num_elements];
        std::fill_n(data, num_elements, 0);
        odla_BindToArgument(value, data, shared_data);
      }
      for(auto& value : shared_data->comp->output_values){
        std::size_t num_elements = 1;
        for(auto& shape : value->tensor_info.shape())
          num_elements *= shape;
        float* data = new float[num_elements];
        std::fill_n(data, num_elements, 0);
        odla_BindToOutput(value, data, shared_data);
      }
    }
  }
  _odla_pipeline_zero* ctx = new _odla_pipeline_zero(SingleComp::get_instance()->get_comp());
  ctx->shared_data = shared_data;

  std::cout << "-----------------> created an empty input/output context: " << ctx << std::endl;
  return ctx;
}

void ContextQueues::put(odla_context ctx)
{
  std::cout << "ContextQueues::put -> ctx: " << ctx << std::endl;
  std::lock_guard<std::mutex> guard(write_mutex);
  write_queue->push(ctx);
}

odla_context ContextQueues::get_input_context()
{
  if(nullptr != input_ctx){
    return input_ctx;
  }
  input_ctx = read_queue->front();
  if( nullptr == input_ctx)
  {
    std::lock_guard<std::mutex> guard(write_mutex);
    std::queue<odla_context>* tmp = read_queue;
    read_queue = write_queue;
    write_queue = tmp;
    input_ctx = read_queue->front();
    std::cout << "===============> switched the read write queue, now read queu size is: " << read_queue->size() << std::endl;
  }
  if(nullptr == input_ctx)  //create a zero data if there's not data in the 2 queues
    input_ctx = create_empty_odla_context();
  
  return input_ctx;
}

odla_context ContextQueues::get_output_context()
{
  output_ctx = wait_output_queue.front();
  if(nullptr == output_ctx)
    std::cerr << "No context in the queue when an output gotten" << std::endl; //严重错误了，会导致数据匹配补上了，是不是可以考虑把输入的数据也放到输出里面，比较一下MD5来确保对应关系
  return output_ctx;
}

#define CB_NULL_QUEUE 100
#define CB_NULL_TENSOR 101

popart::StepIOCallback::InputCallback input_callback =
    [&](popart::TensorId id, bool prefetch) -> popart::ConstVoidData {
  popart::logging::info("input callback called {}", id);
  (void)prefetch;
  //Get the data from the context queues
  popart::ConstVoidData data;
  odla_context ctx = ContextQueues::get_instance()->get_input_context();
  if(nullptr != ctx)
  {
    std::cout << "input_callback got ctx: " << ctx << std::endl;
    popart::IArray* p_array = ctx->get_data_by_tensor_id(id);
    if(nullptr != p_array)
    {
      data.data = p_array->data();
      std::cout << "input_callback -> the p_array dataType is: " << p_array->dataType() << ", shape is: " << p_array->shape() << std::endl;
      data.info = popart::TensorInfo(p_array->dataType(), p_array->shape());
    }
    else
    {
      std::cerr << "input_callback -> Can not find the tensor with id: " << id << " in ctx: " << ctx << std::endl;
      exit(CB_NULL_TENSOR);
    }
  }
  else  //the ctx should never be nullptr
  {
    std::cout << "input_callback -> Queue is empty when try to get" << std::endl;
    std::cout << "input_callback -> return nullptr data" << std::endl;
    exit(CB_NULL_QUEUE);
  }
  return data;
};

popart::StepIOCallback::InputCompleteCallback input_complete_callback =
    [&](popart::TensorId id) -> void {
  std::cout << "input_complete_callback -> called: " << id << std::endl;
  odla_context ctx = ContextQueues::get_instance()->get_input_context();
  if(nullptr == ctx)
  {
    std::cout << "input_complete_callback -> Queue is empty when try to get" << std::endl;
    exit(CB_NULL_QUEUE);
  }
  if(ctx->all_tensors_visited()){
    std::cout << "input_complete_callback -> All tensors read for current context:" << ctx << std::endl;
    ContextQueues::get_instance()->all_tensor_read();
  }
};

popart::StepIOCallback::OutputCallback output_callback =
    [&](popart::TensorId id) -> popart::MutableVoidData {
  popart::logging::info("output callback called {}", id);
  std::cout << "output_callback -> called with id: " << id << std::endl;
  odla_context ctx = ContextQueues::get_instance()->get_output_context();
  if(nullptr == ctx)
  {
    std::cout << "output_callback <- Queue is empty when try to get" << std::endl;
    exit(CB_NULL_QUEUE);
  }
  popart::IArray* p_array = ctx->write_data_by_tensor_id(id);
  if(NULL == p_array)
  {
    std::cerr << "output_callback <- Can not find the tensor with id: " << id << " in ctx: " << ctx << std::endl;
    exit(CB_NULL_TENSOR);
  }
  popart::MutableVoidData data;
  data.data = p_array->data();
  data.info = popart::TensorInfo(p_array->dataType(), p_array->shape());
  return data;
};

popart::StepIOCallback::OutputCompleteCallback output_complete_callback =
    [&](popart::TensorId id) -> void {
  popart::logging::info("output complete callback called {}", id);
  std::cout << "output_complete_callback -> called with id: " << id << std::endl;
  odla_context ctx = ContextQueues::get_instance()->get_output_context();
  if(nullptr == ctx)
  {
    std::cout << "output_complete_callback -> Queue is empty when try to get" << std::endl;
    exit(CB_NULL_QUEUE);
  }
  if(ctx->all_tensors_written()){
    std::cout << "output_complete_callback -> All tensors written for current context waiting output: " << ctx << std::endl;
    ContextQueues::get_instance()->all_tensor_written();
    ctx->clear_visited_and_written();
    odla_context temp_ctx = nullptr;
    if(ctx->deletable()){
        std::cout << "Delete the context after notify: " << ctx << std::endl;
        temp_ctx = ctx;
    }
    ctx->notify();  //unblock the request
    if(temp_ctx)
      delete temp_ctx;
  }
};

void pipeline_loop(odla_computation comp)
{
  SingleComp::get_instance()->init_comp("pipeline", 2, 10);
  std::cout << "=============> current comp is: " << comp << std::endl;
  //setup the stepio with allbacks
  popart::StepIOCallback stepio(input_callback,
                              input_complete_callback,
                              output_callback,
                              output_complete_callback);
  int i=0;
  while(!SingleComp::get_instance()->is_done()){
    std::cout << "This is the " << i++ << " time for the inference" << std::endl;
    if(i == INT_MAX)
      i = 0;
    comp->session->run(stepio);
  }
  std::cout << "The pipeline loop finished" << std::endl;
}

std::unique_ptr<popart::SessionOptions> Pipeline::sessionOptions() {
  std::cout << "---> Pipeline::sessionOptions()" << std::endl;
  auto opts =
      std::unique_ptr<popart::SessionOptions>(new popart::SessionOptions());
  opts->enablePipelining = true;
  opts->virtualGraphMode = popart::VirtualGraphMode::Manual;
  std::cout << "<--- Pipeline::sessionOptions()" << std::endl;
  return opts;
}

void Pipeline::compute(odla_computation comp, odla_context context,
                                  odla_compute_mode mode,odla_device device) 
{
  std::cout << "---> Pipeline::compute()" << std::endl;
  ContextQueues::get_instance()->put(context);
  context->wait();
  std::cout << "<--- Pipeline::compute()" << std::endl;
}

std::unique_ptr<popart::SessionOptions> Sequence::sessionOptions() {
  std::cout << "---> Sequence::sessionOptions()" << std::endl;
  auto opts =
      std::unique_ptr<popart::SessionOptions>(new popart::SessionOptions());
  opts->virtualGraphMode = popart::VirtualGraphMode::Auto;
  opts->enableStochasticRounding = true;
  std::cout << "<--- Sequence::sessionOptions()" << std::endl;
  return opts;
}

void Sequence::compute(odla_computation comp, odla_context context,
                                odla_compute_mode mode, odla_device device) 
{
  SingleComp::get_instance()->init_comp("Sequence", 1, 1);
  std::lock_guard<std::mutex> comp_guard(SingleComp::get_instance()->comp_mutex);
  std::cout << "---> Sequence::compute()" << std::endl;
  // Config StepIO
  std::map<popart::TensorId, popart::IArray&> inputs;
  for (auto& input : context->inputs) {
    inputs.emplace(input.first, *input.second);
  }
  std::map<popart::TensorId, popart::IArray&> outputs;
  for (auto& output : context->outputs) {
    outputs.emplace(output.first, *output.second);
  }

  popart::StepIO stepio(inputs, outputs);
  // Run on ipu
  comp->session->run(stepio);
  std::cout << "<--- Sequence::compute()" << std::endl;
}

std::unique_ptr<popart::SessionOptions> MultiThread::sessionOptions() {
  std::cout << "---> Pipeline::sessionOptions()" << std::endl;
  auto opts =
      std::unique_ptr<popart::SessionOptions>(new popart::SessionOptions());
  opts->virtualGraphMode = popart::VirtualGraphMode::Auto;
  opts->enableStochasticRounding = true;
  std::cout << "<--- Pipeline::sessionOptions()" << std::endl;
  return opts;
}