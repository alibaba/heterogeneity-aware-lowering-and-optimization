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
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <condition_variable>

#define g_comp _odla_computation::instance()
// enum ExecutionMode {PIPELINE, PARALLEL, SEQUENCE};

class Execution {
 public:
  Execution() {}
  ~Execution() {}
  virtual void compute(odla_computation comp, odla_context context,
                       odla_compute_mode mode, odla_device device) = 0;
};

class Sequence : public Execution {
 public:
  Sequence() {}
  ~Sequence() {}
  virtual void compute(odla_computation comp, odla_context context,
                       odla_compute_mode mode, odla_device device);
private:
  std::mutex sequence_mutex;  //As global only has one sequence object, so we can use this mutex
};

class Parallel : public Execution{
public:
    virtual void compute(odla_computation comp, odla_context context,
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
              const std::string& n, bool set_pipeline = true);
};

struct _odla_computation {
  std::unique_ptr<popart::Builder> builder;
  std::unique_ptr<popart::InferenceSession> session;
  std::shared_ptr<popart::DeviceInfo> device;
  popart::SessionOptions m_session_opts;
  std::unordered_map<std::string, odla_value> inputs_map;
  std::unordered_map<std::string, odla_value> outputs_map;
  std::vector<odla_value> input_values;
  std::vector<odla_value> output_values;
  target_opts opts;

  // new members for pipeline
  static _odla_computation* m_instance;
  static _odla_computation* instance(){return m_instance;}
  bool m_done;
  std::mutex m_init_mutex;
  Execution* m_executor;

  _odla_computation();
  
  void init();
  bool is_done(){return m_done;}
  void mark_done(){m_done = true;}
  void set_pipeline_stage(const popart::TensorId &nodeOutputName, const std::string& name);
  void set_pipeline_stage(const std::set<popart::TensorId> &nodeOutputNames, const std::string& name);
  void set_session_opts();
  void set_executor();
  void set_opts();
  Execution* executor(){return m_executor;}
};

struct _odla_context {
  odla_computation comp;
  std::map<popart::TensorId, std::unique_ptr<popart::IArray>> inputs;
  std::map<popart::TensorId, std::unique_ptr<popart::IArray>> outputs;
  _odla_context(odla_computation c): comp(c) {}
  virtual void wait() {}
  virtual void notify() {}
  virtual popart::IArray* get_data_by_tensor_id(popart::TensorId id){
    auto iter = inputs.find(id);
    return (inputs.end() == iter) ? NULL : &(*iter->second);
  }
  virtual popart::IArray* write_data_by_tensor_id(popart::TensorId id){
    auto iter = outputs.find(id);
    return (outputs.end() == iter) ? NULL : &(*iter->second);
  }
  virtual bool all_tensors_visited(){return true;}
  virtual bool all_tensors_written(){return true;}
  virtual void clear_visited_and_written(){}
  virtual bool deletable(){return false;}
};
#endif
