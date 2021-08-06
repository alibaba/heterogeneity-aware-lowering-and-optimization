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

#include <mutex>
#include <queue>
#include <chrono>
#include <ODLA/odla.h>
#include <popart/tensorinfo.hpp>
#include <popart/voiddata.hpp>

#include "ODLA/odla_common.h"
#include "common.h"
#include "odla_popart.h"
#include "odla_pipeline.h"
#include "popart_config.h"

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
      std::cout << "Creating the shared data for the _odla_pipeline_empty_context" << std::endl;
      shared_data = new _odla_context(_odla_computation::instance());
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
  _odla_pipeline_empty_context* ctx = new _odla_pipeline_empty_context(_odla_computation::instance());
  ctx->shared_data = shared_data;

  std::cout << "-----------------> created an empty input/output context: " << ctx << std::endl;
  return ctx;
}

ContextQueues* ContextQueues::get_instance() {
  if (nullptr == p_context_queues) {
    std::lock_guard<std::mutex> guard(instance_mutex);
    if (nullptr == p_context_queues) {
      std::cout << "Creating the ContextQueues singleton" << std::endl;
      p_context_queues = new ContextQueues();
      std::cout << "ContextQueues created, starting the pipeline thread"
                << std::endl;
      std::thread pipeline_thread(pipeline_loop,
                                  _odla_computation::instance());
      std::cout << "Pipeline loop started" << std::endl;
      pipeline_thread.detach();
    }
  }
  return p_context_queues;
}

void ContextQueues::put(odla_context ctx)
{
  std::cout << "ContextQueues::put -> ctx: " << ctx << std::endl;
  {
    std::lock_guard<std::mutex> guard(write_mutex);
    write_queue->push(ctx);
    write_wait_queue->push(ctx);  //put the ctx to input & wait_output queue in same order.
  }
  ctx->wait();  //block the request on the queue to wait for output
}

odla_context ContextQueues::get_input_context()
{
  if(nullptr != input_ctx){
    return input_ctx;
  }
  if(!read_queue->empty())
    input_ctx = read_queue->front();
  else  //read queue is empty, switch it
  {
    std::lock_guard<std::mutex> guard(write_mutex);
    std::queue<odla_context>* tmp = read_queue;
    read_queue = write_queue;
    write_queue = tmp;
    std::cout << "===============> switched the read write queue, now read queu size is: " << read_queue->size() << std::endl;
    if(!read_queue->empty())
      input_ctx = read_queue->front();
    else{  //create a zero data if there's not data in the 2 queues
      input_ctx = create_empty_odla_context();
      write_wait_queue->push(input_ctx);  //Make it wait for the return for the empty data
    }
  }

  return input_ctx;
}

odla_context ContextQueues::get_output_context()
{
  if(output_ctx != nullptr)
    return output_ctx;
  //output_ctx = wait_output_queue.front();
  if(!read_wait_queue->empty())
    output_ctx = read_wait_queue->front();
  else{
    //switch the wait queue
    std::lock_guard<std::mutex> guard(write_mutex); //Use the same mutex to save 1 mutex lock for every put
    std::queue<odla_context>* tmp = read_wait_queue;
    read_wait_queue = write_wait_queue;
    write_wait_queue = tmp;
    std::cout << "===============> switched the read write wait queue, now read queu size is: " << read_wait_queue->size() << std::endl;
  }
  if(!read_wait_queue->empty())
      output_ctx = read_wait_queue->front();
  if(nullptr == output_ctx)
    std::cerr << " *** FATAL ERROR *** No context in the queue when an output gotten" << std::endl; //严重错误了，会导致数据匹配补上了，是不是可以考虑把输入的数据也放到输出里面，比较一下MD5来确保对应关系
  return output_ctx;
}

void ContextQueues::all_tensor_read() {
  std::cout << "ContextQueues::all_tensor_read(), ctx: " << input_ctx
            << " poped, and put into wait_output_queue" << std::endl;
  if(!input_ctx->deletable()) //Only pop the non zero ctx, the zero one not in the queue
      read_queue->pop();
  //wait_output_queue.push(input_ctx); //不用push了，进入的时候大家按顺序进入两个Queue
  input_ctx = nullptr;
}

void ContextQueues::all_tensor_written() { //Never delete a context here, only operate on the queue
  //wait_output_queue.pop();
  if(!read_wait_queue->empty()) //There must be an element when all tensor written
    read_wait_queue->pop(); //pop the first one from the read wait queue
  else{ 
    std::cerr << "*** FATAL ERROR *** when all_tensor_written, there is not a ctx in read_wait_queue" << std::endl;
    exit(-1);
  }
  output_ctx = nullptr;
}

#define CB_NULL_QUEUE 100
#define CB_NULL_TENSOR 101

popart::StepIOCallback::InputCallback input_callback =
    [&](popart::TensorId id, bool prefetch) -> popart::ConstVoidData {
  popart::logging::info("input callback called {}", id);
  std::cout << "input_callback called with id: " << id << std::endl;
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
    if(ctx->all_tensors_visited()){
      std::cout << "input_complete_callback -> All tensors read for current context:" << ctx << std::endl;
      ContextQueues::get_instance()->all_tensor_read();
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
  std::cout << "input_complete_callback -> called: " << id << ", nothing to do..." << std::endl;
  // odla_context ctx = ContextQueues::get_instance()->get_input_context();
  // if(nullptr == ctx)
  // {
  //   std::cout << "input_complete_callback -> Queue is empty when try to get" << std::endl;
  //   exit(CB_NULL_QUEUE);
  // }
  // if(ctx->all_tensors_visited()){
  //   std::cout << "input_complete_callback -> All tensors read for current context:" << ctx << std::endl;
  //   ContextQueues::get_instance()->all_tensor_read();
  // }
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