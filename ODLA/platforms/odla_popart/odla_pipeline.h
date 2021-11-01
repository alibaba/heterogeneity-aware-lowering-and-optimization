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

#include <atomic>
#include <chrono>
#include <mutex>
#include <popart/stepio.hpp>
#include <queue>
#include <thread>

#include "ODLA/odla_common.h"
#include "common.h"
#include "odla_popart.h"

void pipeline_loop(odla_computation comp);

class Queue {
 public:
  virtual void init(std::size_t capacity) = 0;
  virtual void put(odla_context ctx) = 0;
  virtual odla_context get_input_context() = 0;
  virtual odla_context get_ctx_by_tensor(const popart::TensorId& id) = 0;
  virtual odla_context get_output_context() = 0;
  virtual void pop_input(odla_context ctx) = 0;
  virtual void pop_output(odla_context ctx) = 0;
  virtual std::size_t size() = 0;
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

  ~ContextQueues() {}
  void init(std::size_t capacity) final {}
  void put(odla_context ctx) final;
  odla_context get_input_context() final;
  odla_context get_ctx_by_tensor(const popart::TensorId& id) final {
    return nullptr;
  }
  odla_context get_output_context() final;
  void pop_input(odla_context ctx) final;
  void pop_output(odla_context ctx) final;
  std::size_t size() final {
    return input_queue_1.size() + input_queue_2.size();
  }
};

class LockFreeQueue : public Queue {
 private:
  std::atomic<odla_context>* buffer_;
  std::size_t capacity_;
  std::uint32_t head_;
  std::atomic<uint32_t> tail_;
  std::uint32_t wait_;
  std::map<popart::TensorId, std::uint32_t> tensor_to_idx_;

 public:
  LockFreeQueue();
  ~LockFreeQueue() {
    if (buffer_) delete[] buffer_;
  }
  void init(std::size_t capacity);
  void put(odla_context ctx) final;
  odla_context get_input_context() final;
  odla_context get_ctx_by_tensor(const popart::TensorId& id) final;
  odla_context get_output_context() final;
  void pop_input(odla_context ctx) final;
  void pop_output(odla_context ctx) final;
  std::size_t size() final {
    return (tail_.load() - wait_ + capacity_) % capacity_;
  }
};

class QManager {
 private:
  Queue* queue_;
  odla_status status_;
  QManager() : queue_(nullptr), status_(ODLA_SUCCESS) {}
  ~QManager() {}
  std::mutex create_mutex_;
  static QManager* instance_;

 public:
  void createQ(std::string queueType);
  void deleteQ();
  inline Queue* getQ() { return queue_; }
  inline void set_status(odla_status status) { status_ = status; }
  inline odla_status get_status() { return status_; }
  static inline QManager* instance() { return instance_; }
};

struct _odla_pipeline_context : public _odla_context {
  _odla_pipeline_context(odla_computation c)
      : _odla_context(c), visited(0), written(0), got_output(false) {}

  std::mutex context_mutex;
  std::condition_variable context_cv;
  std::set<popart::TensorId>
      tensors_visited; // This is the tensor visited by callback
  std::set<popart::TensorId>
      tensors_written; // Record the output tensor written by callback
  int visited;
  int written;
  bool got_output;
  std::chrono::time_point<std::chrono::steady_clock> start;
  std::chrono::time_point<std::chrono::steady_clock> end;
  inline void wait() override {
    while (!got_output) { // wait forever for the output
      if ( ODLA_SUCCESS != QManager::instance()->get_status() )
		  break; // stop wait if we got exception status
      std::unique_lock<std::mutex> lock(context_mutex);
      context_cv.wait_for(lock, std::chrono::milliseconds(100));
    }
    got_output = false; // reset the flag incase context reused.
  }
  inline void notify() override {
    std::unique_lock<std::mutex> lock(context_mutex);
    got_output = true;
    context_cv.notify_one();
  }
  inline popart::IArray* get_data_by_tensor_id(popart::TensorId id) override {
    visited++;
    return &(*(inputs[id]));
  }
  inline popart::IArray* write_data_by_tensor_id(popart::TensorId id) override {
    if (written == 0) {
      end = std::chrono::steady_clock::now();
      std::chrono::duration<float, std::milli> elapsed_ms = end - start;
      popart::logging::info("ONE_REQUEST for ctx: {} took: {} ms", this,
                            elapsed_ms.count());
    }
    written++;
    return &(*(outputs[id]));
  }
  inline bool all_tensors_visited() override {
    if (inputs.size() != comp->input_values.size()) {
      popart::logging::err(
          "ctx {} inputs.size() is {}, does not match graph inputs size {}",
          this, inputs.size(), comp->input_values.size());
      throw std::runtime_error(
          "input size of context did not match the graph inputs size.");
    }
    if (visited == inputs.size()) {
      start = std::chrono::steady_clock::now();
      return true;
    }
    return false;
  }
  inline bool all_tensors_written() override {
    if (outputs.size() != comp->output_values.size()) {
      popart::logging::err(
          "ctx {} outputs.size() is {}, does not match graph outputs size {}",
          this, outputs.size(), comp->output_values.size());
      throw std::runtime_error(
          "output size of context did not match the graph outputs size.");
    }
    if (written == outputs.size()) {
      return true;
    }
    return false;
  }
  inline void clear_visited_and_written() override {
    visited = 0;
    written = 0;
  }
};

struct _odla_pipeline_empty_context : public _odla_pipeline_context {
  odla_context shared_data = nullptr;
  _odla_pipeline_empty_context(odla_computation c)
      : _odla_pipeline_context(c) {}
  inline void wait() override {}
  inline void notify() override {}
  inline bool deletable() override { return true; }
  inline popart::IArray* get_data_by_tensor_id(popart::TensorId id) override {
    if (!shared_data) return nullptr;
    visited++;
    return shared_data->get_data_by_tensor_id(id);
  }
  inline popart::IArray* write_data_by_tensor_id(popart::TensorId id) override {
    if (!shared_data) return nullptr;
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
