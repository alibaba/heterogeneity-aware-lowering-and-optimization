//===- odla_ops_math.cc ---------------------------------------------------===//
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

#include "common.h"
#include "odla_popart.h"

#if !defined(ODLA_VERSION_NUMBER) || (ODLA_VERSION_NUMBER < 50)
#error This library requires minimum ODLA version 0.5
#endif

extern thread_local odla_computation g_comp;

odla_value odla_Abs(odla_value input, const odla_value_id value_id) {
  const auto& name =
      value_id ? std::string(reinterpret_cast<const char*>(value_id)) : "Abs";

  popart::TensorId result =
      g_comp->builder->aiOnnxOpset10().abs({input->tensor_id});
  return new _odla_value(result,
                         {g_comp->builder->getTensorDataType(result),
                          g_comp->builder->getTensorShape(result)},
                         name);
}

odla_value odla_And(odla_value lhs, odla_value rhs,
                    const odla_value_id value_id) {
  const auto& name =
      value_id ? std::string(reinterpret_cast<const char*>(value_id)) : "And";

  popart::TensorId result = g_comp->builder->aiOnnxOpset10().logical_and(
      {lhs->tensor_id, rhs->tensor_id});
  return new _odla_value(result,
                         {g_comp->builder->getTensorDataType(result),
                          g_comp->builder->getTensorShape(result)},
                         name);
}

odla_value odla_ArgMin(odla_value input, odla_int32 axis, odla_bool keep_dims,
                       odla_bool return_last_index,
                       odla_value_type output_value_type,
                       const odla_value_id value_id) {
  const auto& name = value_id
                         ? std::string(reinterpret_cast<const char*>(value_id))
                         : "Argmin";

  axis = axis >= 0 ? axis : input->tensor_info.rank() + axis;
  popart::TensorId reduced = g_comp->builder->aiOnnxOpset10().argmin(
      {input->tensor_id}, axis, keep_dims);

  auto builder = g_comp->builder.get();
  odla_value result = new _odla_value(
      reduced,
      {builder->getTensorDataType(reduced), builder->getTensorShape(reduced)},
      name);
  return result;
}

odla_value odla_Ceil(odla_value input, const odla_value_id value_id) {
  const auto& name =
      value_id ? std::string(reinterpret_cast<const char*>(value_id)) : "Ceil";

  popart::TensorId result =
      g_comp->builder->aiOnnxOpset10().ceil({input->tensor_id});
  return new _odla_value(result,
                         {g_comp->builder->getTensorDataType(result),
                          g_comp->builder->getTensorShape(result)},
                         name);
}

odla_value odla_Clamp(odla_value input, odla_float32 lo, odla_float32 hi,
                      const odla_value_id value_id) {
  const auto& name =
      value_id ? std::string(reinterpret_cast<const char*>(value_id)) : "Clamp";
  // in popart api, 'hi' is in the front of 'lo'
  popart::TensorId result =
      g_comp->builder->aiOnnxOpset10().clip({input->tensor_id}, hi, lo);
  return new _odla_value(result,
                         {g_comp->builder->getTensorDataType(result),
                          g_comp->builder->getTensorShape(result)},
                         name);
}

odla_value odla_Equal(odla_value lhs, odla_value rhs,
                      const odla_value_id value_id) {
  const auto& name =
      value_id ? std::string(reinterpret_cast<const char*>(value_id)) : "Equal";
  popart::TensorId result =
      g_comp->builder->aiOnnxOpset10().equal({lhs->tensor_id, rhs->tensor_id});
  return new _odla_value(result,
                         {g_comp->builder->getTensorDataType(result),
                          g_comp->builder->getTensorShape(result)},
                         name);
}

odla_value odla_Exp(odla_value input, const odla_value_id value_id) {
  const auto& name =
      value_id ? std::string(reinterpret_cast<const char*>(value_id)) : "Exp";
  popart::TensorId result =
      g_comp->builder->aiOnnxOpset10().exp({input->tensor_id});
  return new _odla_value(result,
                         {g_comp->builder->getTensorDataType(result),
                          g_comp->builder->getTensorShape(result)},
                         name);
}

odla_value odla_Greater(odla_value lhs, odla_value rhs,
                        const odla_value_id value_id) {
  const auto& name = value_id
                         ? std::string(reinterpret_cast<const char*>(value_id))
                         : "Greater";
  popart::TensorId result = g_comp->builder->aiOnnxOpset10().greater(
      {lhs->tensor_id, rhs->tensor_id});
  return new _odla_value(result,
                         {g_comp->builder->getTensorDataType(result),
                          g_comp->builder->getTensorShape(result)},
                         name);
}

odla_value odla_Less(odla_value lhs, odla_value rhs,
                     const odla_value_id value_id) {
  const auto& name =
      value_id ? std::string(reinterpret_cast<const char*>(value_id)) : "Less";
  popart::TensorId result =
      g_comp->builder->aiOnnxOpset10().less({lhs->tensor_id, rhs->tensor_id});
  return new _odla_value(result,
                         {g_comp->builder->getTensorDataType(result),
                          g_comp->builder->getTensorShape(result)},
                         name);
}

odla_value odla_Log(odla_value input, const odla_value_id value_id) {
  const auto& name =
      value_id ? std::string(reinterpret_cast<const char*>(value_id)) : "Log";
  popart::TensorId result =
      g_comp->builder->aiOnnxOpset10().log({input->tensor_id});
  return new _odla_value(result,
                         {g_comp->builder->getTensorDataType(result),
                          g_comp->builder->getTensorShape(result)},
                         name);
}

odla_value odla_Max(odla_value lhs, odla_value rhs,
                    const odla_value_id value_id) {
  const auto& name =
      value_id ? std::string(reinterpret_cast<const char*>(value_id)) : "Max";
  popart::TensorId result =
      g_comp->builder->aiOnnxOpset10().max({lhs->tensor_id, rhs->tensor_id});
  return new _odla_value(result,
                         {g_comp->builder->getTensorDataType(result),
                          g_comp->builder->getTensorShape(result)},
                         name);
}

odla_value odla_Min(odla_value lhs, odla_value rhs,
                    const odla_value_id value_id) {
  const auto& name =
      value_id ? std::string(reinterpret_cast<const char*>(value_id)) : "Min";
  popart::TensorId result =
      g_comp->builder->aiOnnxOpset10().min({lhs->tensor_id, rhs->tensor_id});
  return new _odla_value(result,
                         {g_comp->builder->getTensorDataType(result),
                          g_comp->builder->getTensorShape(result)},
                         name);
}

odla_value odla_Mean(odla_values inputs, const odla_value_id value_id) {
  const auto& name =
      value_id ? std::string(reinterpret_cast<const char*>(value_id)) : "Mean";
  std::vector<popart::TensorId> vec_tensor_id;
  for (int i = 0; i < inputs.size; ++i) {
    vec_tensor_id.emplace_back(inputs.values[i]->tensor_id);
  }
  popart::TensorId result =
      g_comp->builder->aiOnnxOpset10().mean(vec_tensor_id);
  return new _odla_value(result,
                         {g_comp->builder->getTensorDataType(result),
                          g_comp->builder->getTensorShape(result)},
                         name);
}

odla_value odla_Neg(odla_value input, const odla_value_id value_id) {
  const auto& name =
      value_id ? std::string(reinterpret_cast<const char*>(value_id)) : "Neg";
  popart::TensorId result =
      g_comp->builder->aiOnnxOpset10().neg({input->tensor_id});
  return new _odla_value(result,
                         {g_comp->builder->getTensorDataType(result),
                          g_comp->builder->getTensorShape(result)},
                         name);
}

odla_value odla_Not(odla_value input, const odla_value_id value_id) {
  const auto& name =
      value_id ? std::string(reinterpret_cast<const char*>(value_id)) : "Not";
  popart::TensorId result =
      g_comp->builder->aiOnnxOpset10().logical_not({input->tensor_id});
  return new _odla_value(result,
                         {g_comp->builder->getTensorDataType(result),
                          g_comp->builder->getTensorShape(result)},
                         name);
}

odla_value odla_Or(odla_value lhs, odla_value rhs,
                   const odla_value_id value_id) {
  const auto& name =
      value_id ? std::string(reinterpret_cast<const char*>(value_id)) : "Or";

  popart::TensorId result = g_comp->builder->aiOnnxOpset10().logical_or(
      {lhs->tensor_id, rhs->tensor_id});
  return new _odla_value(result,
                         {g_comp->builder->getTensorDataType(result),
                          g_comp->builder->getTensorShape(result)},
                         name);
}

odla_value odla_Pow(odla_value base, odla_value exponent,
                    const odla_value_id value_id) {
  const auto& name =
      value_id ? std::string(reinterpret_cast<const char*>(value_id)) : "Pow";
  popart::TensorId result = g_comp->builder->aiOnnxOpset10().pow(
      {base->tensor_id, exponent->tensor_id});
  return new _odla_value(result,
                         {g_comp->builder->getTensorDataType(result),
                          g_comp->builder->getTensorShape(result)},
                         name);
}

odla_value odla_Reciprocal(odla_value input, const odla_value_id value_id) {
  const auto& name = value_id
                         ? std::string(reinterpret_cast<const char*>(value_id))
                         : "Reciprocal";
  popart::TensorId result =
      g_comp->builder->aiOnnxOpset10().reciprocal({input->tensor_id});
  return new _odla_value(result,
                         {g_comp->builder->getTensorDataType(result),
                          g_comp->builder->getTensorShape(result)},
                         name);
}

odla_value odla_ReduceMax(odla_value input, odla_size_t num_of_axes,
                          const odla_uint32* axes, odla_bool keep_dims,
                          odla_value_shape output_dims,
                          const odla_value_id value_id) {
  const auto& name = value_id
                         ? std::string(reinterpret_cast<const char*>(value_id))
                         : "ReduceMax";

  std::vector<int64_t> axes_vec;
  for (int64_t i = 0; i < num_of_axes; i++) {
    axes_vec.push_back(axes[i]);
  }
  popart::TensorId reduced = g_comp->builder->aiOnnxOpset10().reducemax(
      {input->tensor_id}, axes_vec, keep_dims);

  auto builder = g_comp->builder.get();
  odla_value result = new _odla_value(
      reduced,
      {builder->getTensorDataType(reduced), builder->getTensorShape(reduced)},
      name);
  return result;
}

odla_value odla_ReduceMin(odla_value input, odla_size_t num_of_axes,
                          const odla_uint32* axes, odla_bool keep_dims,
                          odla_value_shape output_dims,
                          const odla_value_id value_id) {
  const auto& name = value_id
                         ? std::string(reinterpret_cast<const char*>(value_id))
                         : "ReduceMin";

  std::vector<int64_t> axes_vec;
  for (int64_t i = 0; i < num_of_axes; i++) {
    axes_vec.push_back(axes[i]);
  }
  popart::TensorId reduced = g_comp->builder->aiOnnxOpset10().reducemin(
      {input->tensor_id}, axes_vec, keep_dims);

  auto builder = g_comp->builder.get();
  odla_value result = new _odla_value(
      reduced,
      {builder->getTensorDataType(reduced), builder->getTensorShape(reduced)},
      name);
  return result;
}

odla_value odla_ReduceProd(odla_value input, odla_size_t num_of_axes,
                           const odla_uint32* axes, odla_bool keep_dims,
                           odla_value_shape output_dims,
                           const odla_value_id value_id) {
  const auto& name = value_id
                         ? std::string(reinterpret_cast<const char*>(value_id))
                         : "ReduceProd";

  std::vector<int64_t> axes_vec;
  for (int64_t i = 0; i < num_of_axes; i++) {
    axes_vec.push_back(axes[i]);
  }
  popart::TensorId reduced = g_comp->builder->aiOnnxOpset10().reduceprod(
      {input->tensor_id}, axes_vec, keep_dims);

  auto builder = g_comp->builder.get();
  odla_value result = new _odla_value(
      reduced,
      {builder->getTensorDataType(reduced), builder->getTensorShape(reduced)},
      name);
  return result;
}

odla_value odla_ReduceSum(odla_value input, odla_size_t num_of_axes,
                          const odla_uint32* axes, odla_bool keep_dims,
                          odla_value_shape output_dims,
                          const odla_value_id value_id) {
  const auto& name = value_id
                         ? std::string(reinterpret_cast<const char*>(value_id))
                         : "ReduceSum";

  std::vector<int64_t> axes_vec;
  for (int64_t i = 0; i < num_of_axes; i++) {
    axes_vec.push_back(axes[i]);
  }
  popart::TensorId reduced = g_comp->builder->aiOnnxOpset10().reducesum(
      {input->tensor_id}, axes_vec, keep_dims);

  auto builder = g_comp->builder.get();
  odla_value result = new _odla_value(
      reduced,
      {builder->getTensorDataType(reduced), builder->getTensorShape(reduced)},
      name);
  return result;
}

odla_value odla_Sign(odla_value input, const odla_value_id value_id) {
  const auto& name =
      value_id ? std::string(reinterpret_cast<const char*>(value_id)) : "Sign";
  popart::TensorId result =
      g_comp->builder->aiOnnxOpset10().sign({input->tensor_id});
  return new _odla_value(result,
                         {g_comp->builder->getTensorDataType(result),
                          g_comp->builder->getTensorShape(result)},
                         name);
}
