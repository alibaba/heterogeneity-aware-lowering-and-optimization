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
#include <atomic>
#include <popart/stepio.hpp>

#include "ODLA/odla_common.h"
#include "common.h"
#include "odla_popart.h"

void pipeline_loop(odla_computation comp);

class Queue
{
public:
  virtual void init(std::size_t capacity) = 0;
  virtual void put(odla_context ctx) = 0;
  virtual odla_context get_input_context() = 0;
  virtual odla_context get_output_context() = 0;
  virtual void pop_input(odla_context ctx) = 0;
  virtual void pop_output(odla_context ctx) = 0;
};

class ContextQueues : public Queue {
private:
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

public:
  ContextQueues()
    : read_queue(&input_queue_1),
      write_queue(&input_queue_2),
      read_wait_queue(&wait_output_queue_1),
      write_wait_queue(&wait_output_queue_2),
      input_ctx(nullptr),
      output_ctx(nullptr) {}

  ~ContextQueues(){}
  void init(std::size_t capacity) final{}
  void put(odla_context ctx) final;
  odla_context get_input_context() final;
  odla_context get_output_context() final;
  void pop_input(odla_context ctx) final;
  void pop_output(odla_context ctx) final;
};

class LockFreeQueue : public Queue
{
private:
  //odla_context* buffer_;
  std::atomic<odla_context>* buffer_;
  std::size_t capacity_;
  uint32_t head_;
  std::atomic<uint32_t> tail_;
  uint32_t wait_;
public:
  LockFreeQueue();
  ~LockFreeQueue(){if(buffer_) delete[] buffer_;}
  void init(std::size_t capacity);
  void put(odla_context ctx) final;
  odla_context get_input_context() final;
  odla_context get_output_context() final;
  void pop_input(odla_context ctx) final;
  void pop_output(odla_context ctx) final;
};

class QManager
{
private:
  Queue* queue_;
  QManager():queue_(nullptr){}
  ~QManager(){}
  std::mutex create_mutex_;
  static QManager* instance_;
public:
  void createQ(std::string queueType);
  inline Queue* getQ(){return queue_;}
  static inline QManager* instance(){return instance_;}
};

struct _odla_pipeline_context : public _odla_context {
  _odla_pipeline_context(odla_computation c) : _odla_context(c), visited(0), written(0) {}

  std::mutex context_mutex;
  std::condition_variable context_cv;
  std::set<popart::TensorId>
      tensors_visited; // This is the tensor visited by callback
  std::set<popart::TensorId>
      tensors_written; // Record the output tensor written by callback
  int visited;
  int written;
  inline void wait() override {
    std::unique_lock<std::mutex> lock(context_mutex);
    context_cv.wait(lock);
  }
  inline void notify() override {
    std::unique_lock<std::mutex> lock(context_mutex);
    context_cv.notify_one();
  }
  inline popart::IArray* get_data_by_tensor_id(popart::TensorId id) override {
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
  inline popart::IArray* write_data_by_tensor_id(popart::TensorId id) override {
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
  inline bool all_tensors_visited() override {
    //return (tensors_visited.size() == inputs.size());
    std::cout << "all_tensor_visited() in _odla_pipeline_context called with visited: " << visited << std::endl;
    return (visited == inputs.size());
  }
  inline bool all_tensors_written() override {
    //return (tensors_written.size() == outputs.size());
    return (written == outputs.size());
  }
  inline void clear_visited_and_written() override {
    //tensors_visited.clear();
    //tensors_written.clear();
    visited=0;
    written=0;
  }
};

struct _odla_pipeline_empty_context : public _odla_pipeline_context {
  odla_context shared_data = nullptr;
  _odla_pipeline_empty_context(odla_computation c) : _odla_pipeline_context(c) {}
  inline void wait() override {}
  inline void notify() override {}
  inline bool deletable() override{return true;}
  inline popart::IArray* get_data_by_tensor_id(popart::TensorId id) override {
    if(!shared_data)
      return nullptr;
    visited++;
    return shared_data->get_data_by_tensor_id(id);
  }
  inline popart::IArray* write_data_by_tensor_id(popart::TensorId id) override {
    if(!shared_data)
      return nullptr;
    written++;
    return shared_data->write_data_by_tensor_id(id);
  }
  inline bool all_tensors_visited() override {
    return (visited == shared_data->inputs.size());
  }
  inline bool all_tensors_written() override {
    return (written == shared_data->outputs.size());
  }
};

extern popart::StepIOCallback::InputCallback input_callback;
extern popart::StepIOCallback::InputCompleteCallback input_complete_callback;
extern popart::StepIOCallback::OutputCallback output_callback;
extern popart::StepIOCallback::OutputCompleteCallback output_complete_callback;

#endif // ODLA_PIPELINE_H_