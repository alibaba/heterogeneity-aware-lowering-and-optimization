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
  // std::map<popart::TensorId, std::unique_ptr<popart::IArray>> inputs;
  // std::map<popart::TensorId, std::unique_ptr<popart::IArray>> outputs;
  std::unique_ptr<popart::InferenceSession> session;        //Session变成是图的资源，一个图对应一个设备，一个session，context里面是每次不同的数据
  std::unordered_map<std::string, odla_value> inputs_map;
  std::unordered_map<std::string, odla_value> outputs_map;
  std::vector<odla_value> input_values;
  std::vector<odla_value> output_values;
  target_opts opts;

  _odla_computation(std::unique_ptr<popart::Builder> b)
      : builder(std::move(b)), opts({false, 1, 1}) {}
};

#define ODLA_PIPELINE
struct _odla_context {
  odla_computation comp;
  //
  //所有的数据都应该是和session绑定的，computation只定义图相关的，以及执行环境相关的，和数据无关
  std::map<popart::TensorId, std::unique_ptr<popart::IArray>> inputs;
  std::map<popart::TensorId, std::unique_ptr<popart::IArray>> outputs;
  _odla_context(odla_computation c)
      : comp(c) {}
#ifdef ODLA_PIPELINE  
  std::mutex context_mutex;
  std::condition_variable context_cv;
  std::set<popart::TensorId> tensors_visited; //This is the tensor visited by callback
  std::set<popart::TensorId> tensors_written;  //Record the output tensor written by callback
  //std::unordered_map<std::string, odla_value> inputs_map;
  //std::unordered_map<std::string, odla_value> outputs_map;
  //std::vector<odla_value> input_values;
  //std::vector<odla_value> output_values;

  // _odla_context(odla_computation c,
  //               std::unique_ptr<popart::InferenceSession> sess)
  //     : comp(c), session(std::move(sess)) {}
  void wait()
  {
    std::unique_lock<std::mutex> lock(context_mutex);
    context_cv.wait(lock);
  }
  void notify()
  {
    std::unique_lock<std::mutex> lock(context_mutex);
    context_cv.notify_one();
  }
  popart::IArray* get_data_by_tensor_id(popart::TensorId id){
    auto visited = tensors_visited.find(id);
    if(tensors_visited.end() != visited){
      std::cerr << "get_data_by_tensor_id() -> Multiple callback on the same tensor:" << id << std::endl;
      return NULL;
    }
    auto iter = inputs.find(id);
    if(inputs.end() == iter)
      return NULL;
    else
    {
      tensors_visited.insert(id);
      return &(*iter->second);
    }
  }
  popart::IArray* write_data_by_tensor_id(popart::TensorId id){
    auto written = tensors_written.find(id);
    if(tensors_written.end() != written){
      std::cerr << "write_data_by_tensor_id -> Multiple output callback on the same tensor:" << id << std::endl;
      return NULL;
    }
    auto iter = outputs.find(id);
    if(outputs.end() == iter)
      return NULL;
    else
    {
      tensors_written.insert(id);
      return &(*iter->second);
    }
  }
  bool all_tensors_visited(){
    return (tensors_visited.size() == inputs.size());
  }
  bool all_tensors_written(){
    return (tensors_written.size() == outputs.size());
  }
  void clear_visited_and_written(){
    std::cout << "For test, clear the visited and written record for the context reusing." << std::endl;
    tensors_visited.clear();
    tensors_written.clear();
  }
#endif  
};

#endif
