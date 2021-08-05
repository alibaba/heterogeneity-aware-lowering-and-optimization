//===- odla_pipeline.h
//------------------------------------------------------===//
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

#ifndef ODLA_PIPELINE_H_
#define ODLA_PIPELINE_H_
#include <ODLA/odla.h>
#include <chrono>
#include <mutex>
#include <queue>
#include <thread>

#include "ODLA/odla_common.h"
#include "common.h"
#include "odla_popart.h"

void pipeline_loop(odla_computation comp);

struct ContextQueues {
  std::queue<odla_context> input_queue_1;
  std::queue<odla_context> input_queue_2;
  std::queue<odla_context> wait_output_queue_1;
  std::queue<odla_context> wait_output_queue_2;
  std::mutex write_mutex;
  std::queue<odla_context>* read_queue;
  std::queue<odla_context>* write_queue;
  std::queue<odla_context>* read_wait_queue;
  std::queue<odla_context>* write_wait_queue;
  odla_context input_ctx;  // the context which is under reading
  odla_context output_ctx; // the context which is under writing

  ContextQueues()
      : read_queue(&input_queue_1),
        write_queue(&input_queue_2),
        read_wait_queue(&wait_output_queue_1),
        write_wait_queue(&wait_output_queue_2),
        input_ctx(nullptr),
        output_ctx(nullptr) {}

  void put(odla_context ctx);
  odla_context get_input_context();
  odla_context get_output_context();

  void all_tensor_read() {
    std::cout << "ContextQueues::all_tensor_read(), ctx: " << input_ctx
              << " poped, and put into wait_output_queue" << std::endl;
    if(!input_ctx->deletable()) //Only pop the non zero ctx, the zero one not in the queue
        read_queue->pop();
    //wait_output_queue.push(input_ctx); //不用push了，进入的时候大家按顺序进入两个Queue
    input_ctx = nullptr;
  }
  void all_tensor_written() { //Never delete a context here, only operate on the queue
    //wait_output_queue.pop();
    if(!read_wait_queue->empty()) //There must be an element when all tensor written
      read_wait_queue->pop(); //pop the first one from the read wait queue
    else{ 
      std::cerr << "*** FATAL ERROR *** when all_tensor_written, there is not a ctx in read_wait_queue" << std::endl;
      exit(-1);
    }
    output_ctx = nullptr;
  }

  static ContextQueues* p_context_queues;
  static std::mutex instance_mutex;
  static ContextQueues* get_instance() {
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
};

struct _odla_pipeline : public _odla_context { // re-name it to _odla_pipeline_context
  _odla_pipeline(odla_computation c) : _odla_context(c), visited(0), written(0) {}

  std::mutex context_mutex;
  std::condition_variable context_cv;
  std::set<popart::TensorId>
      tensors_visited; // This is the tensor visited by callback
  std::set<popart::TensorId>
      tensors_written; // Record the output tensor written by callback
  int visited;
  int written;
  virtual void wait() override {
    std::unique_lock<std::mutex> lock(context_mutex);
    context_cv.wait(lock);
  }
  virtual void notify() override {
    std::unique_lock<std::mutex> lock(context_mutex);
    context_cv.notify_one();
  }
  virtual popart::IArray* get_data_by_tensor_id(popart::TensorId id) {
    // auto visited = tensors_visited.find(id);
    // if (tensors_visited.end() != visited) {
    //   std::cerr
    //       << "get_data_by_tensor_id() -> Multiple callback on the same tensor:"
    //       << id << std::endl;
    //   //return NULL;
    // }
    // auto iter = inputs.find(id);
    // if (inputs.end() == iter)
    //   return NULL;
    // else {
    //   tensors_visited.insert(id);
    //   return &(*iter->second);
    // }
    visited++;
    return &(*(inputs[id]));
  }
  virtual popart::IArray* write_data_by_tensor_id(popart::TensorId id) {
    // auto written = tensors_written.find(id);
    // if (tensors_written.end() != written) {
    //   std::cerr << "write_data_by_tensor_id -> Multiple output callback on the "
    //                "same tensor:"
    //             << id << std::endl;
    //   return NULL;
    // }
    // auto iter = outputs.find(id);
    // if (outputs.end() == iter)
    //   return NULL;
    // else {
    //   tensors_written.insert(id);
    //   return &(*iter->second);
    // }
    written++;
    return &(*(outputs[id]));
  }
  virtual bool all_tensors_visited() {
    //return (tensors_visited.size() == inputs.size());
    return (visited == inputs.size());
  }
  virtual bool all_tensors_written() {
    //return (tensors_written.size() == outputs.size());
    return (written == outputs.size());
  }
  virtual void clear_visited_and_written() {
    std::cout << "clear_visited_and_written() -> clear the visited and written "
                 "record for the context reusing."
              << std::endl;
    //tensors_visited.clear();
    //tensors_written.clear();
    visited=0;
    written=0;
  }
};

struct _odla_pipeline_zero : public _odla_pipeline {
  odla_context shared_data = nullptr;
  _odla_pipeline_zero(odla_computation c) : _odla_pipeline(c) {}
  virtual void wait() {}
  virtual void notify() {}
  virtual bool deletable(){return true;}
  virtual popart::IArray* get_data_by_tensor_id(popart::TensorId id) {
    if(!shared_data)
      return nullptr;
    return shared_data->get_data_by_tensor_id(id);
  }
  virtual popart::IArray* write_data_by_tensor_id(popart::TensorId id) {
    if(!shared_data)
      return nullptr;
    return shared_data->write_data_by_tensor_id(id);
  }
};
#endif // ODLA_PIPELINE_H_