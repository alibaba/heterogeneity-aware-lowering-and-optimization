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
#include <thread>
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

QManager* QManager::instance_ = new QManager();

static odla_context shared_data = nullptr;
static std::mutex shared_data_mutext;

odla_context create_empty_odla_context()
{
  if(!shared_data){
    std::lock_guard<std::mutex> guard(shared_data_mutext);
    if(!shared_data){
      popart::logging::info(
        "Creating the shared data for the _odla_pipeline_empty_context");
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
  _odla_pipeline_empty_context* ctx = 
    new _odla_pipeline_empty_context(_odla_computation::instance());
  ctx->shared_data = shared_data;

  popart::logging::info("created an empty input/output for context: {}.", ctx);
  return ctx;
}

void QManager::createQ(std::string queueType)
{
  if(nullptr == queue_){
    std::lock_guard<std::mutex> guard(create_mutex_);
    if(nullptr == queue_){
      if(queueType == "ContextQueues")
        queue_ = new ContextQueues();
      else if(queueType == "LockFreeQueue")
        queue_ = new LockFreeQueue();
      else
        throw std::invalid_argument("[QManager::createQ] invalid queueType: "
                + queueType + ", should be ContextQueues or LockFreeQueue.");
      popart::logging::info("Created queue with queueType: {}.", queueType);
    }
  }
}

void ContextQueues::put(odla_context ctx)
{
  popart::logging::info("ContextQueues::put -> ctx: {}.", ctx);
  {
    std::lock_guard<std::mutex> guard(write_mutex);
    write_queue->push(ctx);
    write_wait_queue->push(ctx);  //put the ctx to input & wait_output queue in same order.
  }
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
    popart::logging::info(
      "switched the read write queue, now read queu size is: {}.", 
      read_queue->size());
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
  if(!read_wait_queue->empty())
    output_ctx = read_wait_queue->front();
  else{
    //switch the wait queue
    std::lock_guard<std::mutex> guard(write_mutex); //Use the same mutex to save 1 mutex lock for every put
    std::queue<odla_context>* tmp = read_wait_queue;
    read_wait_queue = write_wait_queue;
    write_wait_queue = tmp;
    popart::logging::info(
      "switched the read write wait queue, now read queu size is: {}.", 
      read_wait_queue->size());
  }
  if(!read_wait_queue->empty())
      output_ctx = read_wait_queue->front();
  if(nullptr == output_ctx)
    throw std::out_of_range(
      "*** FATAL ERROR *** No context in the queue when an output gotten");
  return output_ctx;
}

void ContextQueues::pop_input() {
  popart::logging::info("ContextQueues::pop_input with ctx: {}", input_ctx);
  if(!input_ctx->deletable()) //Only pop the non zero ctx, the zero one not in the queue
      read_queue->pop();
  input_ctx = nullptr;
}

void ContextQueues::pop_output() { //Never delete a context here, only operate on the queue
  //wait_output_queue.pop();
  if(!read_wait_queue->empty()) //There must be an element when all tensor written
    read_wait_queue->pop(); //pop the first one from the read wait queue
  else{ 
    throw std::out_of_range(
      "*** FATAL ERROR *** no ctx in read_wait_queue when pop_output called");
  }
  output_ctx = nullptr;
}

/*------------------------------------------------------------------------*/
LockFreeQueue::LockFreeQueue():head_(0), tail_(0), wait_(0)
{
}

void LockFreeQueue::init(std::size_t capacity)
{
  buffer_ = new odla_context[capacity];
  if(nullptr == buffer_)
    throw std::invalid_argument(
      "LockFreeQueue::init failed to create buffer for queue with capacity : "
       + std::to_string(capacity));
  capacity_ = capacity;
}

void LockFreeQueue::put(odla_context ctx)
{
  int idx = 0;
  int new_idx = 0;
  int cnt = 0;
  popart::logging::info(
    "[LockFreeQueue::put] Finding a place to put ctx: {}", ctx);
  do{
    cnt++;
    idx = tail_;
    new_idx = (idx+1) % capacity_;
  }while(!tail_.compare_exchange_strong(idx, new_idx));
  if(new_idx == wait_) // last item as the boundary
    throw std::out_of_range("[LockFreeQueue::put] the queue is full");
  popart::logging::info(
    "[LockFreeQueue::put] Got the idx: {} for ctx: {} in {} times.", 
    idx, ctx, cnt);
  buffer_[idx] = ctx;
}

odla_context LockFreeQueue::get_input_context() //read是callback单线程操作，并且不能马上出队，需要所有tensor都读完
{
  int i = 0;
  while(head_ == tail_.load())
  {
    assert(i++<2);  //only loop once, it must find an element, otherwise wrong
    popart::logging::info(
      "[get_input_context] the queue is empty when read, add zero contexts");
    odla_context zero_ctx = create_empty_odla_context();
    put(zero_ctx);
    popart::logging::info(
      "After this we expect at least 4 read on this ctx: {}.", zero_ctx);
  }
  return buffer_[head_];
}

odla_context LockFreeQueue::get_output_context()
{
  if(wait_ == tail_.load())
    throw std::out_of_range(
      "[LockFreeQueue] queue is empty when get_output_context().");
  return buffer_[wait_];
}

void LockFreeQueue::pop_input()
{
  popart::logging::info("pop_input called with ctx: {}", buffer_[head_]);
  head_ = (head_+1) % capacity_;
}

void LockFreeQueue::pop_output()
{
  wait_ = (wait_ + 1) % capacity_;
}


/*======================================== step io callbacks =========================================*/
popart::StepIOCallback::InputCallback input_callback =
    [&](popart::TensorId id, bool prefetch) -> popart::ConstVoidData {
  // auto start = std::chrono::steady_clock::now();
  
  odla_context ctx = QManager::instance()->getQ()->get_input_context();
  popart::logging::info("InputCallback called with id: {}, ctx: {}", id, ctx);
  popart::IArray* p_array = ctx->get_data_by_tensor_id(id);
  popart::ConstVoidData data(
    p_array->data(),
    popart::TensorInfo(p_array->dataType(), p_array->shape()));
  if(ctx->all_tensors_visited()){
    QManager::instance()->getQ()->pop_input();
  }

  // auto end = std::chrono::steady_clock::now();
  // std::chrono::duration<double> elapsed_seconds = end-start;
  // popart::logging::info(
  //   "[EYECATCHER] ctx: {} popart::StepIOCallback::InputCallback takes {} s."
  //   , ctx, elapsed_seconds.count());
  return data;
};

popart::StepIOCallback::InputCompleteCallback input_complete_callback =
    [&](popart::TensorId id) -> void {
};

popart::StepIOCallback::OutputCallback output_callback =
    [&](popart::TensorId id) -> popart::MutableVoidData {
  // auto start = std::chrono::steady_clock::now();

  odla_context ctx = QManager::instance()->getQ()->get_output_context();
  popart::logging::info("OutputCallback called with id: {} ctx: {}", id, ctx);
  popart::IArray* p_array = ctx->write_data_by_tensor_id(id);
  popart::MutableVoidData data;
  data.data = p_array->data();
  data.info = popart::TensorInfo(p_array->dataType(), p_array->shape());

  // auto end = std::chrono::steady_clock::now();
  // std::chrono::duration<double> elapsed_seconds = end-start;
  // popart::logging::info(
  //   "[EYECATCHER] ctx: {} popart::StepIOCallback::OutputCallback takes {} s.",
  //   ctx, elapsed_seconds.count());
  return data;
};

popart::StepIOCallback::OutputCompleteCallback output_complete_callback =
    [&](popart::TensorId id) -> void {
  // auto start = std::chrono::steady_clock::now();
  odla_context ctx = QManager::instance()->getQ()->get_output_context();
  popart::logging::info(
    "OutputCompleteCallback called with id: {}, ctx: {}", id, ctx);
  if(ctx->all_tensors_written()){
    popart::logging::info("All tensors written for ctx: {}", ctx);
    QManager::instance()->getQ()->pop_output();
    ctx->clear_visited_and_written();
    if(ctx->deletable()){
        ctx->notify();  //unblock the request
        popart::logging::info("ctx: {} has been deleted", ctx);
    }
    else
      ctx->notify();  //unblock the request
  }

  // auto end = std::chrono::steady_clock::now();
  // std::chrono::duration<double> elapsed_seconds = end-start;
  // popart::logging::info(
  //   "[EYECATCHER] ctx: {} OutputCompleteCallback takes {} s.", 
  //   ctx, elapsed_seconds.count());
};