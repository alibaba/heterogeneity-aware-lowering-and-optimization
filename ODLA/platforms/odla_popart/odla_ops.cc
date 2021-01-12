//===- odla_ops.cc --------------------------------------------------------===//
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

/* Ops */
/* Binary Ops */
odla_value odla_Add(odla_value lhs, const odla_value rhs,
                    const odla_value_id id) {
  const auto& name = id ? std::string(reinterpret_cast<const char*>(id)) : "";
  popart::TensorId result =
      g_comp->builder->aiOnnxOpset10().add({lhs->tensor_id, rhs->tensor_id});
  return new _odla_value(result,
                         {g_comp->builder->getTensorDataType(result),
                          g_comp->builder->getTensorShape(result)},
                         name);
}

odla_value odla_Sub(odla_value lhs, odla_value rhs, const odla_value_id id) {
  const auto& name = id ? std::string(reinterpret_cast<const char*>(id)) : "";

  popart::TensorId result =
      g_comp->builder->aiOnnxOpset10().sub({lhs->tensor_id, rhs->tensor_id});
  return new _odla_value(result,
                         {g_comp->builder->getTensorDataType(result),
                          g_comp->builder->getTensorShape(result)},
                         name);
}

odla_value odla_Mul(odla_value lhs, odla_value rhs, const odla_value_id id) {
  const auto& name = id ? std::string(reinterpret_cast<const char*>(id)) : "";

  popart::TensorId result =
      g_comp->builder->aiOnnxOpset10().mul({lhs->tensor_id, rhs->tensor_id});
  return new _odla_value(result,
                         {g_comp->builder->getTensorDataType(result),
                          g_comp->builder->getTensorShape(result)},
                         name);
}

odla_value odla_Div(odla_value lhs, odla_value rhs, const odla_value_id id) {
  const auto& name = id ? std::string(reinterpret_cast<const char*>(id)) : "";

  popart::TensorId result =
      g_comp->builder->aiOnnxOpset10().div({lhs->tensor_id, rhs->tensor_id});
  return new _odla_value(result,
                         {g_comp->builder->getTensorDataType(result),
                          g_comp->builder->getTensorShape(result)},
                         name);
}

/* Unary Ops */
// Erf is declared, but not implemented in onnx namespace,
// so we call the custom version temporialy
odla_value odla_Erf(odla_value input, const odla_value_id id) {
  const auto& name = id ? std::string(reinterpret_cast<const char*>(id)) : "";

  const popart::OperatorIdentifier erf(popart::Domain::ai_graphcore, "Erf", 1,
                                       1, 1);
  auto result = g_comp->builder->customOp(erf, 1, {input->tensor_id}, 1, {})[0];

  return new _odla_value(result,
                         {g_comp->builder->getTensorDataType(result),
                          g_comp->builder->getTensorShape(result)},
                         name);
}

odla_value odla_Floor(odla_value input, const odla_value_id id) {
  const auto& name = id ? std::string(reinterpret_cast<const char*>(id)) : "";

  popart::TensorId result =
      g_comp->builder->aiOnnxOpset10().floor({input->tensor_id});
  return new _odla_value(result,
                         {g_comp->builder->getTensorDataType(result),
                          g_comp->builder->getTensorShape(result)},
                         name);
}

odla_value odla_Sqrt(odla_value input, const odla_value_id id) {
  const auto& name = id ? std::string(reinterpret_cast<const char*>(id)) : "";

  popart::TensorId result =
      g_comp->builder->aiOnnxOpset10().sqrt({input->tensor_id});
  return new _odla_value(result,
                         {g_comp->builder->getTensorDataType(result),
                          g_comp->builder->getTensorShape(result)},
                         name);
}

odla_value odla_Rsqrt(odla_value input, const odla_value_id id) {
  const auto& name = id ? std::string(reinterpret_cast<const char*>(id)) : "";

  const popart::OperatorIdentifier rsqrt(popart::Domain::ai_graphcore, "Rsqrt",
                                         1, 1, 1);
  auto result =
      g_comp->builder->customOp(rsqrt, 1, {input->tensor_id}, 1, {})[0];

  return new _odla_value(result,
                         {g_comp->builder->getTensorDataType(result),
                          g_comp->builder->getTensorShape(result)},
                         name);
}

odla_value odla_Dropout(odla_value input, odla_float32 dropout_prob,
                        const odla_value_id id) {
  const auto& name = id ? std::string(reinterpret_cast<const char*>(id)) : "";

  popart::TensorId result = g_comp->builder->aiOnnxOpset10().dropout(
      {input->tensor_id}, 1, dropout_prob)[0];
  return new _odla_value(result,
                         {g_comp->builder->getTensorDataType(result),
                          g_comp->builder->getTensorShape(result)},
                         name);
}

odla_value odla_Gather(odla_value input, odla_value indices, odla_int32 axis,
                       odla_value_shape output_dims, const odla_value_id id) {
  const auto& name = id ? std::string(reinterpret_cast<const char*>(id)) : "";

  axis = axis >= 0 ? axis : input->tensor_info.rank() + axis;
  popart::TensorId result = g_comp->builder->aiOnnxOpset10().gather(
      {input->tensor_id, indices->tensor_id}, axis);

  return new _odla_value(result,
                         {g_comp->builder->getTensorDataType(result),
                          g_comp->builder->getTensorShape(result)},
                         name);
}

odla_value odla_BatchMatmul(odla_value lhs, odla_bool lhs_trans, odla_value rhs,
                            odla_bool rhs_trans, odla_value_shape output_dims,
                            const odla_value_id id) {
  const auto& name = id ? std::string(reinterpret_cast<const char*>(id)) : "";

  popart::TensorId result =
      g_comp->builder->aiOnnxOpset10().matmul({lhs->tensor_id, rhs->tensor_id});

  return new _odla_value(result,
                         {g_comp->builder->getTensorDataType(result),
                          g_comp->builder->getTensorShape(result)},
                         name);
}

// Y = alpha*A*B + beta*C, popart::Gemm only support A and B rank == 2,
// popart::Matmul support rank range from 1 to 4, but it does not support
// bias and transpose
odla_value odla_Gemm(odla_value lhs, odla_bool transpose_lhs, odla_value rhs,
                     odla_bool transpose_rhs, odla_float32 alpha,
                     odla_float32 beta, odla_value bias,
                     odla_value_shape output_dims, const odla_value_id id) {
  const auto& name = id ? std::string(reinterpret_cast<const char*>(id)) : "";

  popart::TensorId lhs_trans = lhs->tensor_id;
  int rank = lhs->tensor_info.rank();
  if (rank > 2 && transpose_lhs) {
    if (rank == 4) {
      lhs_trans = g_comp->builder->aiOnnxOpset10().transpose(
          {lhs->tensor_id}, std::vector<int64_t>{0, 1, 3, 2});
    } else if (rank == 3) {
      lhs_trans = g_comp->builder->aiOnnxOpset10().transpose(
          {lhs->tensor_id}, std::vector<int64_t>{0, 2, 1});
    }
  }

  popart::TensorId rhs_trans = rhs->tensor_id;
  rank = rhs->tensor_info.rank();
  if (rank > 2 && transpose_rhs) {
    if (rank == 4) {
      rhs_trans = g_comp->builder->aiOnnxOpset10().transpose(
          {rhs->tensor_id}, std::vector<int64_t>{0, 1, 3, 2});
    } else if (rank == 3) {
      rhs_trans = g_comp->builder->aiOnnxOpset10().transpose(
          {rhs->tensor_id}, std::vector<int64_t>{0, 2, 1});
    }
  }

  // USE_BATCHED_MATMUL
  popart::TensorId result =
      g_comp->builder->aiOnnxOpset10().matmul({lhs_trans, rhs_trans});

  return new _odla_value(result,
                         {g_comp->builder->getTensorDataType(result),
                          g_comp->builder->getTensorShape(result)},
                         name);
}

odla_value odla_Reshape(odla_value input, odla_value_shape output_dims,
                        const odla_value_id id) {
  const auto& name = id ? std::string(reinterpret_cast<const char*>(id)) : "";

  // TODO(unknown) memory leak ?
  auto new_shape = odla_CreateConstant(
      {ODLA_INT64, {.size = 1, .dims = {output_dims.size}}},
      static_cast<odla_void*>(GetPopartShape(output_dims).data()),
      (const odla_value_id)((name + "_new_shape").c_str()));

  popart::TensorId result = g_comp->builder->aiOnnxOpset10().reshape(
      {input->tensor_id, new_shape->tensor_id}, name);
  return new _odla_value(result,
                         {g_comp->builder->getTensorDataType(result),
                          g_comp->builder->getTensorShape(result)},
                         name);
}

odla_value odla_OneHot(odla_value indices, odla_int32 depth, odla_value values,
                       odla_int32 axis, odla_value_shape output_dims,
                       const odla_value_id id) {
  const auto& name = id ? std::string(reinterpret_cast<const char*>(id)) : "";

  std::vector<int64_t> depth_data = {depth};
  auto depth_v = odla_CreateConstant(
      {ODLA_INT64, {.size = 1, .dims = {1}}},
      static_cast<odla_void*>(depth_data.data()),
      (const odla_value_id)((name + "_depth_value").c_str()));
  popart::TensorId result = g_comp->builder->aiOnnxOpset10().onehot(
      {indices->tensor_id, depth_v->tensor_id, values->tensor_id});
  return new _odla_value(result,
                         {g_comp->builder->getTensorDataType(result),
                          g_comp->builder->getTensorShape(result)},
                         name);
}

odla_value odla_Relu(odla_value input, const odla_value_id id) {
  const auto& name = id ? std::string(reinterpret_cast<const char*>(id)) : "";

  popart::TensorId result =
      g_comp->builder->aiOnnxOpset10().relu({input->tensor_id});
  return new _odla_value(result,
                         {g_comp->builder->getTensorDataType(result),
                          g_comp->builder->getTensorShape(result)},
                         name);
}

odla_value odla_Softmax(odla_value input, odla_int32 axis,
                        const odla_value_id id) {
  const auto& name = id ? std::string(reinterpret_cast<const char*>(id)) : "";

  axis = axis >= 0 ? axis : input->tensor_info.rank() + axis;
  popart::TensorId result =
      g_comp->builder->aiOnnxOpset10().softmax({input->tensor_id}, axis);
  return new _odla_value(result,
                         {g_comp->builder->getTensorDataType(result),
                          g_comp->builder->getTensorShape(result)},
                         name);
}

odla_value odla_GroupNormalization(odla_value input,
                                   odla_memory_layout input_layout,
                                   odla_value mean, odla_value var,
                                   odla_float32 epsilon, odla_value scale,
                                   odla_value offset, odla_float32 scalar_scale,
                                   odla_float32 scalar_offset,
                                   const odla_value_id id) {
  // TODO(unknown) mean var not in use, check group_norm/batch_norm
  const auto& name = id ? std::string(reinterpret_cast<const char*>(id)) : "";

  auto outs = g_comp->builder->aiGraphcoreOpset1().groupnormalization(
      {input->tensor_id, scale->tensor_id, offset->tensor_id}, 1, epsilon);
  return new _odla_value(outs[0],
                         {g_comp->builder->getTensorDataType(outs[0]),
                          g_comp->builder->getTensorShape(outs[0])},
                         name);
}

odla_value odla_ArgMax(odla_value input, odla_int32 axis, odla_bool keep_dims,
                       odla_bool return_last_index,
                       odla_value_type output_value_type,
                       const odla_value_id id) {
  const auto& name = id ? std::string(reinterpret_cast<const char*>(id)) : "";

  axis = axis >= 0 ? axis : input->tensor_info.rank() + axis;
  popart::TensorId reduced = g_comp->builder->aiOnnxOpset10().argmax(
      {input->tensor_id}, axis, keep_dims);

  auto builder = g_comp->builder.get();
  odla_value result = new _odla_value(
      reduced,
      {builder->getTensorDataType(reduced), builder->getTensorShape(reduced)},
      name);
  return result;
}

odla_value odla_ReduceMean(odla_value input, odla_size_t num_of_axes,
                           const odla_uint32* axes, odla_bool keep_dims,
                           odla_value_shape output_dims,
                           const odla_value_id id) {
  const auto& name = id ? std::string(reinterpret_cast<const char*>(id)) : "";

  std::vector<int64_t> axes_vec;
  for (int64_t i = 0; i < num_of_axes; i++) {
    axes_vec.push_back(axes[i]);
  }
  popart::TensorId reduced = g_comp->builder->aiOnnxOpset10().reducemean(
      {input->tensor_id}, axes_vec, keep_dims);

  auto builder = g_comp->builder.get();
  odla_value result = new _odla_value(
      reduced,
      {builder->getTensorDataType(reduced), builder->getTensorShape(reduced)},
      name);
  return result;
}

odla_value odla_Slice(odla_value input, const odla_uint32* start,
                      const odla_uint32* end, const odla_uint32* stride,
                      odla_value_shape output_dims, const odla_value_id id) {
  const auto& name = id ? std::string(reinterpret_cast<const char*>(id)) : "";

  std::vector<int64_t> starts;
  std::vector<int64_t> ends;
  std::vector<int64_t> axes;
  std::vector<int64_t> strides;

  int64_t rank = input->tensor_info.rank();
  for (int64_t i = 0; i < rank; i++) {
    starts.push_back(start[i]);
    ends.push_back(end[i]);
    axes.push_back(i);
    strides.push_back(stride[i]);
  }

  auto start_tensor =
      odla_CreateConstant({ODLA_INT64, {.size = 1, .dims = {rank}}},
                          static_cast<odla_void*>(starts.data()),
                          (const odla_value_id)((name + "_starts").c_str()));

  auto end_tensor =
      odla_CreateConstant({ODLA_INT64, {.size = 1, .dims = {rank}}},
                          static_cast<odla_void*>(ends.data()),
                          (const odla_value_id)((name + "_ends").c_str()));

  auto axes_tensor =
      odla_CreateConstant({ODLA_INT64, {.size = 1, .dims = {rank}}},
                          static_cast<odla_void*>(axes.data()),
                          (const odla_value_id)((name + "_axes").c_str()));

  auto strides_tensor =
      odla_CreateConstant({ODLA_INT64, {.size = 1, .dims = {rank}}},
                          static_cast<odla_void*>(strides.data()),
                          (const odla_value_id)((name + "_strides").c_str()));

  auto builder = g_comp->builder.get();
  popart::TensorId sliced = builder->aiOnnxOpset10().slice(
      {input->tensor_id, start_tensor->tensor_id, end_tensor->tensor_id,
       axes_tensor->tensor_id, strides_tensor->tensor_id});

  odla_value result = new _odla_value(
      sliced,
      {builder->getTensorDataType(sliced), builder->getTensorShape(sliced)},
      name);
  return result;
}

odla_value odla_Transpose(odla_value input, odla_value_shape permutations,
                          odla_value_shape output_dims,
                          const odla_value_id id) {
  const auto& name = id ? std::string(reinterpret_cast<const char*>(id)) : "";

  // If permutations.size is 0, perm is empty.
  // popart reverses the dimensions by default.
  std::vector<int64_t> perm;
  for (int64_t i = 0; i < permutations.size; i++) {
    perm.push_back(permutations.dims[i]);
  }

  popart::TensorId transposed =
      g_comp->builder->aiOnnxOpset10().transpose({input->tensor_id}, perm);

  auto builder = g_comp->builder.get();
  odla_value result = new _odla_value(transposed,
                                      {builder->getTensorDataType(transposed),
                                       builder->getTensorShape(transposed)},
                                      name);
  return result;
}
