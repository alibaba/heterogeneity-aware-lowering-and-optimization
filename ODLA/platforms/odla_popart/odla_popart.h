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

#include <atomic>
#include <condition_variable>
#include <popart/builder.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/session.hpp>
#include <popart/sessionoptions.hpp>
#include <popart/tensorinfo.hpp>
#include <string>
#include <thread>
#include <vector>

#define g_comp _odla_computation::instance()

class Execution {
 public:
  Execution() {}
  ~Execution() {}
  virtual odla_status compute(odla_computation comp, odla_context context,
                              odla_compute_mode mode, odla_device device) = 0;
};

class Sequence : public Execution {
 public:
  Sequence() {}
  ~Sequence() {}
  virtual odla_status compute(odla_computation comp, odla_context context,
                              odla_compute_mode mode, odla_device device);

 private:
  std::mutex sequence_mutex; // As global only has one sequence object, so we
                             // can use this mutex
};

class Parallel : public Execution {
 public:
  virtual odla_status compute(odla_computation comp, odla_context context,
                              odla_compute_mode mode, odla_device device);
};

typedef struct TargetOpts {
  bool use_ipu_model;
  int64_t ipu_num;
  int64_t batches_per_step;
  bool enable_engine_cache;
  const char* cache_dir;
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
  std::unique_ptr<popart::InferenceSession> session;
  std::shared_ptr<popart::DeviceInfo> device;
  popart::SessionOptions session_opts_;
  std::unordered_map<std::string, odla_value> inputs_map;
  std::unordered_map<std::string, odla_value> outputs_map;
  std::vector<odla_value> input_values;
  std::vector<odla_value> output_values;
  target_opts opts;

  // new members for pipeline
  enum THREAD_STATE { RUNNING = 0, MARK_DONE, DONE };
  THREAD_STATE thread_state_;
  std::mutex thread_done_mutex_;
  std::condition_variable thread_done_cv_;

  static _odla_computation* instance_;
  static std::mutex comp_mutex_;
  static _odla_computation* instance(bool hold_it = true) {
    if (instance_ == nullptr) {
      std::lock_guard<std::mutex> guard(comp_mutex_);
      if (instance_ == nullptr) instance_ = new _odla_computation();
      popart::logging::warn("The computation:{} has been firstly created",
                            instance_);
    }
    if (hold_it) instance_->hold();
    return instance_;
  }
  static void destruct() {
    if (instance_ != nullptr) {
      std::lock_guard<std::mutex> guard(comp_mutex_);
      if (instance_ != nullptr) {
        delete instance_;
        popart::logging::warn("The computation:{} has been destructed",
                              instance_);
        instance_ = nullptr;
      }
    }
  }
  bool is_compile_only_;
  bool done_;
  bool thread_complete_;
  std::mutex init_mutex_;
  Execution* executor_;
  std::thread::id thread_id_of_holder;

  _odla_computation()
      : builder(popart::Builder::create()),
        session(nullptr),
        device(nullptr),
        opts({false, 1, 1}),
        done_(false),
        is_compile_only_(false),
        executor_(nullptr),
        thread_state_(DONE) {
    builder->setAttribute(popart::sVirtualGraphAttribute, 0);
  }
  std::string set_pipeline_stage();
  void set_session_opts();

  bool use_pipeline();
  bool hold();

  odla_status init_working_thread();
  odla_status init(bool is_compile = false);
  odla_status set_executor();
  odla_status set_opts();
  odla_status compile_and_export();

  inline Execution* executor() { return executor_; }
  inline bool is_done() { return thread_state_ != RUNNING; }
  inline bool is_compile_only() { return is_compile_only_; }
  inline void release_session() {
    if (session != nullptr) {
      session->getDevice().getDeviceInfo()->detach();
      popart::logging::warn(
          "The computation:{} session:{} detached from device", this,
          session.get());
      session.reset();
      assert(session == nullptr);
      popart::logging::warn("The computation:{} session has been reset", this);
    }
  }
  inline void set_thread_run() {
    std::unique_lock<std::mutex> lock(thread_done_mutex_);
    thread_state_ = RUNNING;
  }
  inline void mark_done() {
    while (thread_state_ != DONE) {
      std::unique_lock<std::mutex> lock(thread_done_mutex_);
      if (thread_state_ != DONE) {
        thread_state_ = MARK_DONE;
        popart::logging::warn(
            "The computation:{} thread now is MARK_DONE, waiting for DONE",
            this);
        thread_done_cv_.wait_for(lock, std::chrono::milliseconds(5));
      } else
        popart::logging::warn(
            "The computation {} thread already DONE when try to mark_done",
            this);
    }
    // Once get notified, only detach the device once
    std::lock_guard<std::mutex> guard(init_mutex_);
    release_session();
  }
  inline void thread_done() {
    std::unique_lock<std::mutex> lock(thread_done_mutex_);
    thread_state_ = DONE;
    popart::logging::warn("The computation:{} thread is DONE.", this);
    thread_done_cv_.notify_all();
  }
};

struct _odla_context {
  odla_computation comp;
  std::map<popart::TensorId, std::unique_ptr<popart::IArray>> inputs;
  std::map<popart::TensorId, std::unique_ptr<popart::IArray>> outputs;
  int (*async_callback_func)(void*, odla_status) = nullptr;
  void* async_callback_arg = nullptr;
  _odla_context(odla_computation c) : comp(c) {}
  std::thread::id thread_id_of_holder;
  inline virtual void wait() {}
  inline virtual void notify() {}
  inline virtual popart::IArray* get_data_by_tensor_id(popart::TensorId id) {
    auto iter = inputs.find(id);
    return (inputs.end() == iter) ? NULL : &(*iter->second);
  }
  inline virtual popart::IArray* write_data_by_tensor_id(popart::TensorId id) {
    auto iter = outputs.find(id);
    return (outputs.end() == iter) ? NULL : &(*iter->second);
  }
  inline virtual bool all_tensors_visited() { return true; }
  inline virtual bool all_tensors_written() { return true; }
  inline virtual void clear_visited_and_written() {}
  inline virtual bool deletable() { return false; }
  virtual bool hold(const std::string& function_name);
};
#endif
