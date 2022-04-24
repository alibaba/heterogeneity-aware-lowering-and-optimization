//===- odla_pipeline.cc
//----------------------------------------------------===//
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

#include "odla_pipeline.h"

#include <ODLA/odla.h>

#include <chrono>
#include <mutex>
#include <popart/tensorinfo.hpp>
#include <popart/voiddata.hpp>
#include <queue>
#include <thread>

#include "ODLA/odla_common.h"
#include "common.h"
#include "odla_popart.h"
#include "popart_config.h"

#if !defined(ODLA_VERSION_NUMBER) || (ODLA_VERSION_NUMBER < 50)
#error This library requires minimum ODLA version 0.5
#endif

QManager* QManager::instance_ = new QManager();

static odla_context shared_data = nullptr;
static std::mutex shared_data_mutext;

odla_context create_empty_odla_context() {
  if (!shared_data) {
    std::lock_guard<std::mutex> guard(shared_data_mutext);
    if (!shared_data) {
      popart::logging::info(
          "Creating the shared data for the _odla_pipeline_empty_context");
      shared_data = new _odla_context(_odla_computation::instance());
      for (auto& value : shared_data->comp->input_values) {
        std::size_t num_elements = 1;
        for (auto& shape : value->tensor_info.shape()) num_elements *= shape;
        float* data = new float[num_elements];
        std::fill_n(data, num_elements, 0);
        odla_BindToArgument(value, data, shared_data);
      }
      for (auto& value : shared_data->comp->output_values) {
        std::size_t num_elements = 1;
        for (auto& shape : value->tensor_info.shape()) num_elements *= shape;
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

void QManager::createQ(std::string queueType) {
  if (nullptr == queue_) {
    std::lock_guard<std::mutex> guard(create_mutex_);
    if (nullptr == queue_) {
      if (queueType == "ContextQueues")
        queue_ = new ContextQueues();
      else if (queueType == "LockFreeQueue")
        queue_ = new LockFreeQueue();
      else
        throw std::invalid_argument(
            "[QManager::createQ] invalid queueType: " + queueType +
            ", should be ContextQueues or LockFreeQueue.");
      status_ = ODLA_SUCCESS; // reset the QManager status when create new queue
      popart::logging::info("Created queue with queueType: {}.", queueType);
    }
  }
}

void QManager::deleteQ() {
  if (nullptr != queue_) {
    std::lock_guard<std::mutex> guard(create_mutex_);
    if (nullptr != queue_) {
      delete queue_;
      queue_ = nullptr;
    }
  }
}

void ContextQueues::init(std::size_t capacity) {
  buffer_ = new odla_context[capacity];
  if (nullptr == buffer_)
    throw std::invalid_argument(
        "ContextQueues::init failed to create buffer for queue with capacity "
        ": " +
        std::to_string(capacity));
  for (int i = 0; i < capacity; i++) buffer_[i] = nullptr;
  capacity_ = capacity;
}

void ContextQueues::put(odla_context ctx) {
  popart::logging::info("ContextQueues::put -> ctx: {}.", ctx);
  {
    std::lock_guard<std::mutex> guard(queue_mutex_);
    auto new_tail = (tail_ + 1) % capacity_;
    if (new_tail == wait_) // last item as the boundary
      throw std::out_of_range("ContextQueues::put the queue is full");
    buffer_[tail_] = ctx;
    tail_ = new_tail;
  } // Make sure the queue mutex released here.
  // Notify the batch wait we got a batch data
  std::unique_lock<std::mutex> lock(batch_wait_mutex_);
  batch_wait_cv_.notify_one();
}

odla_context ContextQueues::get_input_context() {
  throw std::runtime_error(
      "ContextQueues::get_input_context we should never call this.");
}

odla_context ContextQueues::get_output_context() {
  if (wait_ == tail_)
    throw std::out_of_range(
        "ContextQueues: queue is empty when get_output_context()");
  return buffer_[wait_];
}

void ContextQueues::pop_input(odla_context ctx) {
  popart::logging::info("ContextQueues::pop_input with ctx: {}", ctx);
  assert(ctx == buffer_[head_]);
  head_ = (head_ + 1) % capacity_;
}

void ContextQueues::pop_output(odla_context ctx) {
  if (wait_ == head_)
    throw std::runtime_error("Got out before input all read on index " +
                             std::to_string(wait_));
  assert(ctx == buffer_[wait_]);
  buffer_[wait_] = nullptr; // clear the buffer to nullptr;
  wait_ = (wait_ + 1) % capacity_;
}

odla_context ContextQueues::get_ctx_by_tensor(const popart::TensorId& id) {
  std::uint32_t idx = -1;
  odla_context ctx = nullptr;
  // Get current index
  auto iter = tensor_to_idx_.find(id);
  if (tensor_to_idx_.end() == iter)
    idx = 0;
  else
    idx = iter->second;
  // Check whether is empty, tail alwasy points to the first element not written
  std::uint32_t cnt = 0;
  popart::logging::info("ContextQueues::get_ctx_by_tensor queue has size: {}",
                        size());
  while (idx == tail_) {
    auto locked_tail = tail_;
    {
      std::lock_guard<std::mutex> guard(queue_mutex_);
      locked_tail = tail_;
    }
    if (idx == locked_tail) {
      std::unique_lock<std::mutex> lock(batch_wait_mutex_);
      batch_wait_cv_.wait_for(lock, std::chrono::milliseconds(5));
    }
    if (idx != tail_) break;
    popart::logging::info(
        "[get_ctx_by_tensor] the queue is empty when read, add zero contexts");
    if (cnt++ > 1)
      throw std::runtime_error(
          "[get_ctx_by_tensor] Must get one ctx in 2 fetch, as empty one "
          "created.");
    odla_context zero_ctx = create_empty_odla_context();
    put(zero_ctx);
  }
  // The lock ensured the ctx has been written
  ctx = buffer_[idx];
  popart::logging::info(
      "ContextQueues::get_ctx_by_tensor tensorid:{} got ctx:{} with idx: {}",
      id, ctx, idx);
  // Update the index of the tensor to next
  tensor_to_idx_[id] = (idx + 1) % capacity_;
  return ctx;
}

void ContextQueues::handle_error() {
  auto idx = wait_;
  while (idx != tail_) {
    odla_context ctx = buffer_[idx];
    if (ctx != nullptr) ctx->notify();
    idx = (idx + 1) % capacity_;
  }
}

/*------------------------------------------------------------------------*/
LockFreeQueue::LockFreeQueue() : head_(0), tail_(0), wait_(0) {}

void LockFreeQueue::init(std::size_t capacity) {
  buffer_ = new std::atomic<odla_context>[capacity];
  if (nullptr == buffer_)
    throw std::invalid_argument(
        "LockFreeQueue::init failed to create buffer for queue with capacity "
        ": " +
        std::to_string(capacity));
  for (int i = 0; i < capacity; i++) buffer_[i] = nullptr;
  capacity_ = capacity;
}

void LockFreeQueue::put(odla_context ctx) {
  uint32_t idx = 0;
  uint32_t new_idx = 0;
  uint32_t cnt = 0;
  if (ctx == nullptr)
    throw std::invalid_argument("LockFreeQueue::put null ctx received");
  popart::logging::info("[LockFreeQueue::put] Finding a place to put ctx: {}",
                        ctx);
  do {
    cnt++;
    idx = tail_;
    new_idx = (idx + 1) % capacity_;
  } while (!tail_.compare_exchange_strong(idx, new_idx));

  if (new_idx == wait_) // last item as the boundary
    throw std::out_of_range("LockFreeQueue::put the queue is full");
  popart::logging::info(
      "[LockFreeQueue::put] Got the idx: {} for ctx: {} in {} times.", idx, ctx,
      cnt);

  odla_context null_value = nullptr;
  cnt = 0;
  while (!buffer_[idx].compare_exchange_strong(
      null_value, ctx)) { // avoid consumed by the head_ before written
    if (cnt++ > 5)
      throw std::runtime_error("LockFreeQueue::put No one should stop me");
  }
  // Notify the batch wait we got a batch data
  std::unique_lock<std::mutex> lock(batch_wait_mutex_);
  batch_wait_cv_.notify_one();
  popart::logging::info(
      "[LockFreeQueue::put] Set the idx: {} for ctx: {} in {} times.", idx, ctx,
      cnt);
}

odla_context LockFreeQueue::get_input_context() // only return the head_, ++
                                                // will be done when pop_input
{
  std::uint32_t cnt = 0;
  popart::logging::info("LockFreeQueue::get_input_context queue has size: {}",
                        size());
  while (head_ == tail_.load()) {
    popart::logging::info(
        "[get_input_context] the queue is empty when read, add zero contexts");
    if (cnt++ > 1)
      throw std::runtime_error(
          "Must get one ctx in 2 fetch, as empty one created.");
    odla_context zero_ctx = create_empty_odla_context();
    put(zero_ctx);
  }
  cnt = 0;
  while (buffer_[head_].load() == nullptr) {
    if (cnt++ > 666) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    if (cnt++ > 6666)
      throw std::runtime_error("Must get a valid ctx in 6666 times idx: " +
                               std::to_string(head_));
    popart::logging::info(
        "Reading ctx with head_: {} encounter nullptr {} times.", head_, cnt);
  }
  return buffer_[head_].load();
}

odla_context LockFreeQueue::get_output_context() {
  if (wait_ == tail_.load())
    throw std::out_of_range(
        "[LockFreeQueue] queue is empty when get_output_context().");
  return buffer_[wait_].load();
}

void LockFreeQueue::pop_input(odla_context ctx) {
  popart::logging::info("pop_input called with ctx: {}, head_: {}", ctx, head_);
  assert(ctx == buffer_[head_].load());
  head_ = (head_ + 1) % capacity_;
}

void LockFreeQueue::pop_output(odla_context ctx) {
  if (wait_ == head_)
    throw std::runtime_error("Got out before input all read on index " +
                             std::to_string(wait_));
  if (!buffer_[wait_].compare_exchange_strong(ctx, nullptr))
    throw std::runtime_error("pop_output() there should not have competion");
  wait_ = (wait_ + 1) % capacity_;
}

odla_context LockFreeQueue::get_ctx_by_tensor(const popart::TensorId& id) {
  std::uint32_t idx = -1;
  odla_context ctx = nullptr;
  // Get current index
  auto iter = tensor_to_idx_.find(id);
  if (tensor_to_idx_.end() == iter)
    idx = 0;
  else
    idx = iter->second;
  // Check whether is empty, tail alwasy points to the first element not written
  std::uint32_t cnt = 0;
  popart::logging::info("LockFreeQueue::get_ctx_by_tensor queue has size: {}",
                        size());
  while (idx == tail_.load()) {
    if (idx == tail_.load()) {
      std::unique_lock<std::mutex> lock(batch_wait_mutex_);
      batch_wait_cv_.wait_for(lock, std::chrono::milliseconds(5));
    }
    if (idx != tail_.load()) break;
    popart::logging::info(
        "[get_ctx_by_tensor] the queue is empty when read, add zero contexts");
    if (cnt++ > 1)
      throw std::runtime_error(
          "[get_ctx_by_tensor] Must get one ctx in 2 fetch, as empty one "
          "created.");
    odla_context zero_ctx = create_empty_odla_context();
    put(zero_ctx);
  }
  // The idx not equals to tail_, means there should be ctx written, but need to
  // check has it been written?
  cnt = 0;
  while (buffer_[idx].load() == nullptr) {
    if (cnt++ > 666) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    if (cnt++ > 6666)
      throw std::runtime_error(
          "[get_ctx_by_tensor] Must get a valid ctx in 6666 times idx: {}" +
          std::to_string(idx));
    popart::logging::info(
        "[get_ctx_by_tensor] Reading ctx with idx: {} encounter nullptr {} "
        "times.",
        idx, cnt);
  }
  ctx = buffer_[idx].load();
  popart::logging::info(
      "LockFreeQueue::get_ctx_by_tensor tensorid:{} got ctx:{} with idx: {}",
      id, ctx, idx);
  // Update the index of the tensor to next
  tensor_to_idx_[id] = (idx + 1) % capacity_;
  return ctx;
}

void LockFreeQueue::handle_error() {
  auto idx = wait_;
  while (idx != tail_.load()) {
    odla_context ctx = buffer_[idx].load();
    if (ctx != nullptr) ctx->notify();
    idx = (idx + 1) % capacity_;
  }
}

/*======================================== step io callbacks
 * =========================================*/
popart::StepIOCallback::InputCallback input_callback =
    [](popart::TensorId id, bool prefetch) -> popart::ConstVoidData {
  odla_context ctx = QManager::instance()->getQ()->get_ctx_by_tensor(
      id); // get_input_context();
  popart::ConstVoidData data;
  if (nullptr != ctx) {
    popart::logging::info("InputCallback called with id: {}, ctx: {}", id, ctx);
    popart::IArray* p_array = ctx->get_data_by_tensor_id(id);
    data.data = p_array->data();
    data.info = popart::TensorInfo(p_array->dataType(), p_array->shape());
    if (ctx->all_tensors_visited()) {
      QManager::instance()->getQ()->pop_input(ctx);
    }
  } else {
    data.data = nullptr;
  }
  return data;
};

popart::StepIOCallback::InputCompleteCallback input_complete_callback =
    [](popart::TensorId id) -> void {};

popart::StepIOCallback::OutputCallback output_callback =
    [](popart::TensorId id) -> popart::MutableVoidData {
  odla_context ctx = QManager::instance()->getQ()->get_output_context();
  popart::logging::info("OutputCallback called with id: {} ctx: {}", id, ctx);
  popart::IArray* p_array = ctx->write_data_by_tensor_id(id);
  popart::MutableVoidData data;
  data.data = p_array->data();
  data.info = popart::TensorInfo(p_array->dataType(), p_array->shape());

  return data;
};

popart::StepIOCallback::OutputCompleteCallback output_complete_callback =
    [](popart::TensorId id) -> void {
  odla_context ctx = QManager::instance()->getQ()->get_output_context();
  popart::logging::info("OutputCompleteCallback called with id: {}, ctx: {}",
                        id, ctx);
  if (ctx->all_tensors_written()) {
    popart::logging::info("All tensors written for ctx: {}", ctx);
    QManager::instance()->getQ()->pop_output(ctx);
    ctx->clear_visited_and_written();
    if (ctx->deletable()) { // only empty context can be deleted,
      ctx->notify();        // and no one will use the ctx after notify
      delete ctx;
      popart::logging::info("ctx: {} has been deleted", ctx);
    } else
      ctx->notify(); // unblock the request
  }
};
