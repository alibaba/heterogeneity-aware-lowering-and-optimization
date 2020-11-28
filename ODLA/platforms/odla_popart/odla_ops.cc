//
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
#include "odla_popart.h"
#include "common.h"

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
#if 0
// Erf is declared, but not implemented in onnx namespace,
// so we call the custom version temporialy
odla_value odla_Erf(odla_value input, const odla_value_id id) {
  const auto& name = id ? std::string(reinterpret_cast<const char*>(id)) : "";

  popart::TensorId result =
      g_comp->builder->aiOnnxOpset10().erf({input->tensor_id});
  return new _odla_value(result,
                         {g_comp->builder->getTensorDataType(result),
                          g_comp->builder->getTensorShape(result)},
                         name);
}
#endif
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

std::string _VecToStr(const std::vector<int64_t>& vec) {
  std::ostringstream oss;
  oss << "{";
  for (int n = 0; n < vec.size(); ++n) {
    if (n != 0) {
      oss << ", ";
    }
    oss << vec[n];
  }
  oss << "}";
  return oss.str();
}

odla_value odla_Matmul(odla_value lhs, odla_value rhs, const odla_value_id id) {
  const auto& name = id ? std::string(reinterpret_cast<const char*>(id)) : "";

  popart::TensorId result =
      g_comp->builder->aiOnnxOpset10().matmul({lhs->tensor_id, rhs->tensor_id});

 // std::cout << "Found Matmul: " << static_cast<std::string>(result) << std::endl;
 // std::cout << "  lhs: " << static_cast<std::string>(lhs->tensor_id) << std::endl;
 // std::cout << "  lhs data_type: " << g_comp->builder->getTensorDataType(lhs->tensor_id) << std::endl;
 // std::cout << "  lhs shape: " << _VecToStr(g_comp->builder->getTensorShape(lhs->tensor_id)) << std::endl;
 // std::cout << "  rhs: " << static_cast<std::string>(rhs->tensor_id) << std::endl;
 // std::cout << "  rhs data_type: " << g_comp->builder->getTensorDataType(rhs->tensor_id) << std::endl;
 // std::cout << "  rhs shape: " << _VecToStr(g_comp->builder->getTensorShape(rhs->tensor_id)) << std::endl;
 // std::cout << "  result shape: " << _VecToStr(g_comp->builder->getTensorShape(result)) << std::endl;

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
#if 1 // USE_BATCHED_MATMUL
  const auto& name = id ? std::string(reinterpret_cast<const char*>(id)) : "";

 //  std::cout << "transA:" << transpose_lhs << ", transB: " << transpose_rhs
 //            << "\n";
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

  popart::TensorId result =
      g_comp->builder->aiOnnxOpset10().matmul({lhs_trans, rhs_trans});

//  std::cout << "Found Matmul: " << static_cast<std::string>(result)
//            << std::endl;
//  std::cout << "  lhs: " << static_cast<std::string>(lhs->tensor_id)
//            << std::endl;
//  std::cout << "  lhs data_type: "
//            << g_comp->builder->getTensorDataType(lhs->tensor_id) << std::endl;
//  std::cout << "  lhs shape: "
//            << _VecToStr(g_comp->builder->getTensorShape(lhs->tensor_id))
//            << std::endl;
//  std::cout << "  rhs: " << static_cast<std::string>(rhs->tensor_id)
//            << std::endl;
//  std::cout << "  rhs data_type: "
//            << g_comp->builder->getTensorDataType(rhs->tensor_id) << std::endl;
//  std::cout << "  rhs shape: "
//            << _VecToStr(g_comp->builder->getTensorShape(rhs->tensor_id))
//            << std::endl;
//  std::cout << "  result shape: "
//            << _VecToStr(g_comp->builder->getTensorShape(result)) << std::endl;
//
  return new _odla_value(result,
                         {g_comp->builder->getTensorDataType(result),
                          g_comp->builder->getTensorShape(result)},
                         name);
#else // USE_GEMM
  const auto& name = id ? std::string(reinterpret_cast<const char*>(id)) : "";
  std::cout << "======name: " << name << "==========\n";
  std::cout << "transA:" << transpose_lhs << ", transB: " << transpose_rhs
            << "\n";
  popart::TensorId lhs_reshape = lhs->tensor_id;
  if (int rank = lhs->tensor_info.rank(); rank > 2) {
    std::cout << "lhs shape: "
              << g_comp->builder->getTensorShape(lhs->tensor_id)[0] << ", "
              << g_comp->builder->getTensorShape(lhs->tensor_id)[1] << ", "
              << g_comp->builder->getTensorShape(lhs->tensor_id)[2] << ", "
              << g_comp->builder->getTensorShape(lhs->tensor_id)[3] << "\n";
    std::vector<int64_t> shape0{
        -1, lhs->tensor_info.dim(rank - 1) * lhs->tensor_info.dim(rank - 2)};
    auto new_shape = odla_CreateConstant(
        {ODLA_INT64, {.size = 1, .dims = {2}}},
        static_cast<odla_void*>(shape0.data()),
        (const odla_value_id)((name + "_new_shape0").c_str()));
    lhs_reshape = g_comp->builder->aiOnnxOpset10().reshape(
        {lhs->tensor_id, new_shape->tensor_id}, name + "_new_shape0");
    std::cout << "popart lhs shape: "
              << g_comp->builder->getTensorShape(lhs_reshape)[0] << ", "
              << g_comp->builder->getTensorShape(lhs_reshape)[1] << "\n";
  }

  popart::TensorId rhs_reshape = rhs->tensor_id;
  if (int rank = rhs->tensor_info.rank(); rank > 2) {
    std::cout << "rhs shape: "
              << g_comp->builder->getTensorShape(rhs->tensor_id)[0] << ", "
              << g_comp->builder->getTensorShape(rhs->tensor_id)[1] << ", "
              << g_comp->builder->getTensorShape(rhs->tensor_id)[2] << ", "
              << g_comp->builder->getTensorShape(rhs->tensor_id)[3] << "\n";
    std::vector<int64_t> shape1{
        -1, rhs->tensor_info.dim(rank - 1) * rhs->tensor_info.dim(rank - 2)};
    auto new_shape = odla_CreateConstant(
        {ODLA_INT64, {.size = 1, .dims = {2}}},
        static_cast<odla_void*>(shape1.data()),
        (const odla_value_id)((name + "_new_shape1").c_str()));
    rhs_reshape = g_comp->builder->aiOnnxOpset10().reshape(
        {rhs->tensor_id, new_shape->tensor_id}, name + "_new_shape1");
    std::cout << "popart rhs shape: "
              << g_comp->builder->getTensorShape(rhs_reshape)[0] << ", "
              << g_comp->builder->getTensorShape(rhs_reshape)[1] << "\n";
  }

  std::vector<int64_t> bias_data{0};
  auto C = odla_CreateConstant({ODLA_FLOAT32, {.size = 1, .dims = {1}}},
                               static_cast<odla_void*>(bias_data.data()),
                               (const odla_value_id)((name + "_bias").c_str()));
  popart::TensorId result = g_comp->builder->aiOnnxOpset10().gemm(
      {lhs_reshape, rhs_reshape, bias ? bias->tensor_id : C->tensor_id}, alpha,
      beta, static_cast<int64_t>(transpose_lhs),
      static_cast<int64_t>(transpose_rhs), name);

  popart::TensorId result_reshape = result;
  if (output_dims.size > 2) {
    std::cout << "out shape: " << output_dims.dims[0] << ", "
              << output_dims.dims[1] << ", " << output_dims.dims[2] << ", "
              << output_dims.dims[3] << "\n";
    auto new_shape = odla_CreateConstant(
        {ODLA_INT64, {.size = 1, .dims = {output_dims.size}}},
        static_cast<odla_void*>(GetPopartShape(output_dims).data()),
        (const odla_value_id)((name + "_new_shape2").c_str()));
    result_reshape = g_comp->builder->aiOnnxOpset10().reshape(
        {result, new_shape->tensor_id}, name + "_new_shape2");
    std::cout << "popart out shape: "
              << g_comp->builder->getTensorShape(result_reshape)[0] << ", "
              << g_comp->builder->getTensorShape(result_reshape)[1] << ", "
              << g_comp->builder->getTensorShape(result_reshape)[2] << ", "
              << g_comp->builder->getTensorShape(result_reshape)[3] << "\n";
  }

  return new _odla_value(result_reshape,
                         {g_comp->builder->getTensorDataType(result_reshape),
                          g_comp->builder->getTensorShape(result_reshape)},
                         name);
#endif
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
  auto builder = g_comp->builder.get();
  
  auto start_tensor = odla_CreateConstant(
      {ODLA_INT64, {.size = 1, .dims = {rank}}}, static_cast<odla_void*>(starts.data()),
      (const odla_value_id)((name + "_starts").c_str()));
 
  auto end_tensor = odla_CreateConstant(
      {ODLA_INT64, {.size = 1, .dims = {rank}}}, static_cast<odla_void*>(ends.data()),
      (const odla_value_id)((name + "_ends").c_str()));

  auto axes_tensor = odla_CreateConstant(
      {ODLA_INT64, {.size = 1, .dims = {rank}}}, static_cast<odla_void*>(axes.data()),
      (const odla_value_id)((name + "_axes").c_str()));

  auto strides_tensor = odla_CreateConstant(
      {ODLA_INT64, {.size = 1, .dims = {rank}}}, static_cast<odla_void*>(strides.data()),
      (const odla_value_id)((name + "_strides").c_str()));

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
