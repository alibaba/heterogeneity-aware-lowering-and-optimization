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

typedef struct TargetOpts {
  bool use_ipu_model;
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
  std::unique_ptr<popart::InferenceSession> session;
  std::unordered_map<std::string, odla_value> inputs_map;
  std::unordered_map<std::string, odla_value> outputs_map;
  std::vector<odla_value> input_values;
  std::vector<odla_value> output_values;
  target_opts opts;

  _odla_computation(std::unique_ptr<popart::Builder> b)
      : builder(std::move(b)), opts({false, 1, 1}) {}
};

struct _odla_context {
  odla_computation comp;
  std::map<popart::TensorId, std::unique_ptr<popart::IArray>> inputs;
  std::map<popart::TensorId, std::unique_ptr<popart::IArray>> outputs;
  _odla_context(odla_computation c): comp(c) {}
  virtual void wait() {}
  virtual void notify() {}
  virtual popart::IArray* get_data_by_tensor_id(popart::TensorId id) = 0;
  virtual popart::IArray* write_data_by_tensor_id(popart::TensorId id) = 0;
  virtual bool all_tensors_visited(){return true;}
  virtual bool all_tensors_written(){return true;}
  virtual void clear_visited_and_written(){}
  virtual bool deletable(){return false;}
};

struct SingleComp {
  odla_computation single_comp;
  std::mutex initalization_mutex;
  bool comp_initialized = false;
  static std::mutex instance_mutex;
  static SingleComp* instance;
  odla_computation get_comp() { return single_comp; }
  void init_comp();
  static SingleComp* get_instance() {
    if (nullptr == instance) {
      std::lock_guard<std::mutex> guard(instance_mutex);
      if (nullptr == instance) {
        instance = new SingleComp();
        // Create the single computation
        std::unique_ptr<popart::Builder> builder = popart::Builder::create();
        // Place Subgraph on IPU 0
        builder->setAttribute(popart::sVirtualGraphAttribute, 0);
        instance->single_comp = new _odla_computation(std::move(builder));
      }
    }
    return instance;
  }
};
#endif
