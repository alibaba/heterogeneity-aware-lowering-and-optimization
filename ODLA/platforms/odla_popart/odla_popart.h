//===- odla_popart.h ------------------------------------------------------===//
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

#ifndef ODLA_POPART_H_
#define ODLA_POPART_H_

#include <ODLA/odla.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <numeric>
#include <popart/builder.hpp>
#include <popart/dataflow.hpp>
#include <popart/devicemanager.hpp>
#include <popart/names.hpp>
#include <popart/ndarraywrapper.hpp>
#include <popart/session.hpp>
#include <popart/sessionoptions.hpp>
#include <popart/stepio.hpp>
#include <popart/tensordata.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/voiddata.hpp>
#include <queue>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

template <typename T>
class TSQueue {
 public:
  TSQueue() = default;
  TSQueue(const TSQueue<T>&) = delete;
  TSQueue& operator=(const TSQueue<T>&) = delete;
  TSQueue(TSQueue<T>&& other) {
    std::lock_guard<std::mutex> lock(mutex_);
    queue_ = std::move(other.queue_);
  }
  virtual ~TSQueue() {}

  unsigned long size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size();
  }

  T front() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (queue_.empty()) {
      return {};
    }
    return queue_.front();
  }

  T back() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (queue_.empty()) {
      return {};
    }
    return queue_.back();
  }

  void pop() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (queue_.empty()) {
      return;
    }
    queue_.pop();
    return;
  }

  void push(const T& data) {
    std::lock_guard<std::mutex> lock(mutex_);
    queue_.push(data);
  }

  void push(const T&& data) {
    std::lock_guard<std::mutex> lock(mutex_);
    queue_.emplace(data);
  }

  bool empty() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.empty();
  }

  void clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    while (!queue_.empty()) {
      queue_.pop();
    }
  }

 private:
  std::queue<T> queue_;
  mutable std::mutex mutex_;
};

typedef struct TargetOpts {
  bool use_ipu_model;
  bool enable_pipeline;
  int64_t ipu_num;
  int64_t batches_per_step;
} target_opts;

struct _odla_value {
  popart::TensorId tensor_id;
  popart::TensorInfo tensor_info;
  std::string name;

  _odla_value(popart::TensorId id, popart::TensorInfo info,
              const std::string& n)
      : tensor_id(id), tensor_info(info), name(n) {}
};

struct _odla_computation {
  std::unique_ptr<popart::Builder> builder;
  std::map<popart::TensorId, std::unique_ptr<popart::IArray>> inputs;
  std::map<popart::TensorId, std::unique_ptr<popart::IArray>> outputs;
  std::unordered_map<std::string, odla_value> inputs_map;
  std::unordered_map<std::string, odla_value> outputs_map;
  std::vector<odla_value> input_values;
  std::vector<odla_value> output_values;
  popart::StepIOCallback::InputCallback input_callback;
  popart::StepIOCallback::InputCompleteCallback input_complete_callback;
  popart::StepIOCallback::OutputCallback output_callback;
  popart::StepIOCallback::OutputCompleteCallback output_complete_callback;
  TSQueue<popart::ConstVoidData> in_queue;
  TSQueue<odla_value> out_queue;
  target_opts opts;

  _odla_computation(std::unique_ptr<popart::Builder> b)
      : builder(std::move(b)), opts({false, false, 1, 1}) {}
};

struct _odla_context {
  odla_computation comp;
  std::unique_ptr<popart::InferenceSession> session;

  _odla_context(odla_computation c,
                std::unique_ptr<popart::InferenceSession> sess)
      : comp(c), session(std::move(sess)) {}
};

#endif
