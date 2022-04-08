//===- odla_ops_nn.cc -----------------------------------------------------===//
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

#include <popart/builder.hpp>
#include <popart/tensorinfo.hpp>
#include <string>
#include <vector>

#include "common.h"
#include "odla_popart.h"

#if !defined(ODLA_VERSION_NUMBER) || (ODLA_VERSION_NUMBER < 50)
#error This library requires minimum ODLA version 0.5
#endif

odla_value odla_AveragePool(odla_value input, odla_memory_layout input_layout,
                            const odla_uint32* window_dims,
                            const odla_uint32* strides,
                            const odla_uint32* paddings_front,
                            const odla_uint32* paddings_back,
                            odla_value_shape output_dims,
                            const odla_value_id value_id) {
  const auto& name = value_id
                         ? std::string(reinterpret_cast<const char*>(value_id))
                         : "AveragePool";

  int64_t rank = input->tensor_info.rank() - 2;

  std::vector<int64_t> kernel_shape;
  std::vector<int64_t> dim_strides;
  std::vector<int64_t> padding;
  std::vector<int64_t> padding_from_back;
  for (int64_t i = 0; i < rank; i++) {
    kernel_shape.emplace_back(window_dims[i]);
    dim_strides.emplace_back(strides[i]);
    padding.emplace_back(paddings_front[i]);
    padding_from_back.emplace_back(paddings_back[i]);
  }

  padding.insert(padding.end(), padding_from_back.begin(),
                 padding_from_back.end());
  popart::TensorId result = g_comp->builder->aiOnnxOpset10().averagepool(
      {input->tensor_id}, kernel_shape, 0, 0, padding, dim_strides, name);
  return new _odla_value(result,
                         {g_comp->builder->getTensorDataType(result),
                          g_comp->builder->getTensorShape(result)},
                         name);
}

odla_value odla_BatchNormalization(odla_value input,
                                   odla_memory_layout input_layout,
                                   odla_value mean, odla_value var,
                                   odla_float32 epsilon, odla_value scale,
                                   odla_value offset, odla_float32 scalar_scale,
                                   odla_float32 scalar_offset,
                                   const odla_value_id value_id) {
  const auto& name = value_id
                         ? std::string(reinterpret_cast<const char*>(value_id))
                         : "BatchNormalization";

  // the shape of input suppose to be N x C x D1 x D2 ... Dn.
  int rank = input->tensor_info.rank();
  int channel_dim = 1;
  if (rank > 3) {
    channel_dim = g_comp->builder->getTensorShape(input->tensor_id)[1];
  }

  auto input_popart_type = g_comp->builder->getTensorDataType(input->tensor_id);
  odla_element_type input_type = GetOdlaType(input_popart_type);
  std::vector<float> scale_tmp(channel_dim, scalar_scale);
  if (scale == nullptr) {
    scale = odla_CreateConstant(
        {input_type, {.size = 1, .dims = {channel_dim}}}, scale_tmp.data(),
        (const odla_value_id)(name + "_scale").c_str());
  }
  std::vector<float> offset_tmp(channel_dim, scalar_offset);
  if (offset == nullptr) {
    offset = odla_CreateConstant(
        {input_type, {.size = 1, .dims = {channel_dim}}}, offset_tmp.data(),
        (const odla_value_id)(name + "_offset").c_str());
  }

  auto outs = g_comp->builder->aiOnnxOpset10().batchnormalization(
      {input->tensor_id, scale->tensor_id, offset->tensor_id, mean->tensor_id,
       var->tensor_id},
      1, epsilon, 0.9, name);
  return new _odla_value(outs[0],
                         {g_comp->builder->getTensorDataType(outs[0]),
                          g_comp->builder->getTensorShape(outs[0])},
                         name);
}

odla_value odla_Conv(odla_value input, odla_memory_layout input_layout,
                     odla_uint32 group, odla_value kernel,
                     odla_memory_layout kernel_layout,
                     const odla_uint32* strides, const odla_uint32* dilations,
                     const odla_uint32* paddings_front,
                     const odla_uint32* paddings_back, odla_value bias,
                     odla_value_shape output_dims,
                     const odla_value_id value_id) {
  const auto& name =
      value_id ? std::string(reinterpret_cast<const char*>(value_id)) : "Conv";

  int64_t rank = input->tensor_info.rank() - 2;

  std::vector<int64_t> dim_dilations;
  std::vector<int64_t> dim_strides;
  std::vector<int64_t> pads;
  std::vector<int64_t> pads_from_back;
  for (int64_t i = 0; i < rank; i++) {
    dim_dilations.emplace_back(dilations[i]);
    dim_strides.emplace_back(strides[i]);
    pads.emplace_back(paddings_front[i]);
    pads_from_back.emplace_back(paddings_back[i]);
  }

  pads.insert(pads.end(), pads_from_back.begin(), pads_from_back.end());
  std::vector<int64_t> kernel_shape(
      g_comp->builder->getTensorShape(kernel->tensor_id));
  if (kernel_shape.size() > 2) {
    kernel_shape.erase(kernel_shape.begin(), kernel_shape.begin() + 2);
  }

  std::vector<popart::TensorId> inputs = {input->tensor_id, kernel->tensor_id};
  if (bias != nullptr) {
    inputs.push_back(bias->tensor_id);
  }
  auto result = g_comp->builder->aiOnnxOpset10().conv(
      inputs, dim_dilations, group, kernel_shape, pads, dim_strides, name);
  return new _odla_value(result,
                         {g_comp->builder->getTensorDataType(result),
                          g_comp->builder->getTensorShape(result)},
                         name);
}

odla_value odla_DeConv(odla_value input, odla_memory_layout input_layout,
                       odla_uint32 group, odla_value kernel,
                       odla_memory_layout kernel_layout,
                       const odla_uint32* strides, const odla_uint32* dilations,
                       const odla_uint32* paddings_front,
                       const odla_uint32* paddings_back, odla_value bias,
                       odla_value_shape output_dims,
                       const odla_value_id value_id) {
  const auto& name = value_id
                         ? std::string(reinterpret_cast<const char*>(value_id))
                         : "DeConv";

  int64_t rank = input->tensor_info.rank() - 2;

  std::vector<int64_t> dim_dilations;
  std::vector<int64_t> dim_strides;
  std::vector<int64_t> pads;
  std::vector<int64_t> pads_from_back;
  for (int64_t i = 0; i < rank; i++) {
    dim_dilations.emplace_back(dilations[i]);
    dim_strides.emplace_back(strides[i]);
    pads.emplace_back(paddings_front[i]);
    pads_from_back.emplace_back(paddings_back[i]);
  }
  pads.insert(pads.end(), pads_from_back.begin(), pads_from_back.end());
  std::vector<int64_t> kernel_shape(
      g_comp->builder->getTensorShape(kernel->tensor_id));
  if (kernel_shape.size() > 2) {
    kernel_shape.erase(kernel_shape.begin(), kernel_shape.begin() + 2);
  }

  std::vector<popart::TensorId> inputs = {input->tensor_id, kernel->tensor_id};
  if (bias != nullptr) {
    inputs.push_back(bias->tensor_id);
  }
  auto result = g_comp->builder->aiOnnxOpset10().convtranspose(
      {input->tensor_id, kernel->tensor_id}, dim_dilations, group, kernel_shape,
      std::vector<int64_t>(), // output_padding
      std::vector<int64_t>(), // output_shape
      pads, dim_strides, name);
  return new _odla_value(result,
                         {g_comp->builder->getTensorDataType(result),
                          g_comp->builder->getTensorShape(result)},
                         name);
}

odla_value odla_Elu(odla_value input, odla_float32 alpha,
                    const odla_value_id value_id) {
  const auto& name =
      value_id ? std::string(reinterpret_cast<const char*>(value_id)) : "Elu";

  popart::TensorId result =
      g_comp->builder->aiOnnxOpset10().elu({input->tensor_id}, alpha, name);
  return new _odla_value(result,
                         {g_comp->builder->getTensorDataType(result),
                          g_comp->builder->getTensorShape(result)},
                         name);
}

odla_value odla_HardSigmoid(odla_value input, odla_float32 alpha,
                            odla_float32 beta, const odla_value_id value_id) {
  const auto& name = value_id
                         ? std::string(reinterpret_cast<const char*>(value_id))
                         : "HardSigmoid";

  popart::TensorId result = g_comp->builder->aiOnnxOpset10().hardsigmoid(
      {input->tensor_id}, alpha, beta, name);
  return new _odla_value(result,
                         {g_comp->builder->getTensorDataType(result),
                          g_comp->builder->getTensorShape(result)},
                         name);
}

odla_value odla_InstanceNormalization(
    odla_value input, odla_memory_layout input_layout, odla_float32 epsilon,
    odla_value scale, odla_value offset, odla_float32 scalar_scale,
    odla_float32 scalar_offset, const odla_value_id value_id) {
  const auto& name = value_id
                         ? std::string(reinterpret_cast<const char*>(value_id))
                         : "InstanceNormalization";

  // the shape of input suppose to be N x C x D1 x D2 ... Dn.
  int rank = input->tensor_info.rank();
  int channel_dim = 1;
  if (rank > 3) {
    channel_dim = g_comp->builder->getTensorShape(input->tensor_id)[1];
  }

  if (scale == NULL) {
    std::vector<float> scale_tmp(channel_dim, scalar_scale);
    scale = odla_CreateConstant(
        {ODLA_FLOAT32, {.size = 1, .dims = {channel_dim}}}, scale_tmp.data(),
        (const odla_value_id)(name + "_scale").c_str());
  }
  if (offset == NULL) {
    std::vector<float> offset_tmp(channel_dim, scalar_offset);
    offset = odla_CreateConstant(
        {ODLA_FLOAT32, {.size = 1, .dims = {channel_dim}}}, offset_tmp.data(),
        (const odla_value_id)(name + "_offset").c_str());
  }

  auto result = g_comp->builder->aiOnnxOpset10().instancenormalization(
      {input->tensor_id, scale->tensor_id, offset->tensor_id}, epsilon, name);
  return new _odla_value(result,
                         {g_comp->builder->getTensorDataType(result),
                          g_comp->builder->getTensorShape(result)},
                         name);
}

odla_value odla_Gelu(odla_value input, odla_bool use_approx,
                     const odla_value_id value_id) {
  const auto& name =
      value_id ? std::string(reinterpret_cast<const char*>(value_id)) : "Gelu";
  popart::TensorId result =
      g_comp->builder->aiGraphcoreOpset1().gelu({input->tensor_id}, name);
  return new _odla_value(result,
                         {g_comp->builder->getTensorDataType(result),
                          g_comp->builder->getTensorShape(result)},
                         name);
}

odla_value odla_LeakyRelu(odla_value input, odla_float32 alpha,
                          const odla_value_id value_id) {
  const auto& name = value_id
                         ? std::string(reinterpret_cast<const char*>(value_id))
                         : "LeakyRelu";

  popart::TensorId result = g_comp->builder->aiOnnxOpset10().leakyrelu(
      {input->tensor_id}, alpha, name);
  return new _odla_value(result,
                         {g_comp->builder->getTensorDataType(result),
                          g_comp->builder->getTensorShape(result)},
                         name);
}

extern odla_value odla_LogSoftmax(odla_value input, odla_int32 axis,
                                  const odla_value_id value_id) {
  const auto& name = value_id
                         ? std::string(reinterpret_cast<const char*>(value_id))
                         : "LogSoftmax";

  popart::TensorId result = g_comp->builder->aiOnnxOpset10().logsoftmax(
      {input->tensor_id}, axis, name);
  return new _odla_value(result,
                         {g_comp->builder->getTensorDataType(result),
                          g_comp->builder->getTensorShape(result)},
                         name);
}

odla_values odla_LSTM(odla_value input, odla_value_shape weight_dims,
                      odla_value W, odla_value R, odla_value B,
                      odla_uint32 seq_len, odla_int32 hidden_size,
                      odla_rnn_direction direction, odla_rnn_outputs outputs,
                      const odla_value_id value_id) {
  const auto& name =
      value_id ? std::string(reinterpret_cast<const char*>(value_id)) : "LSTM";
  std::string direction_name = GetDirectionName(direction);
  int input_forget = 0;

  auto outs = g_comp->builder->aiOnnxOpset10().lstm(
      {input->tensor_id, W->tensor_id, R->tensor_id, B->tensor_id},
      3,                          // num_outputs
      std::vector<float>(),       // activation_alpha
      std::vector<float>(),       // activation beta
      std::vector<std::string>(), // activations
      nonstd::optional<float>(),  // clip
      direction_name, hidden_size, input_forget, name);
  odla_value value_1 =
      new _odla_value(outs[0],
                      {g_comp->builder->getTensorDataType(outs[0]),
                       g_comp->builder->getTensorShape(outs[0])},
                      name + "0");
  odla_value value_2 =
      new _odla_value(outs[1],
                      {g_comp->builder->getTensorDataType(outs[1]),
                       g_comp->builder->getTensorShape(outs[1])},
                      name + "1");
  odla_value value_3 =
      new _odla_value(outs[2],
                      {g_comp->builder->getTensorDataType(outs[2]),
                       g_comp->builder->getTensorShape(outs[2])},
                      name + "2");
  return std::move(odla_values{3, {value_1, value_2, value_3}});
}

odla_value odla_MaxPool(odla_value input, odla_memory_layout input_layout,
                        const odla_uint32* window_dims,
                        const odla_uint32* strides,
                        const odla_uint32* paddings_front,
                        const odla_uint32* paddings_back,
                        odla_value_shape output_dims,
                        const odla_value_id value_id) {
  const auto& name = value_id
                         ? std::string(reinterpret_cast<const char*>(value_id))
                         : "MaxPool";

  int64_t rank = input->tensor_info.rank() - 2;

  std::vector<int64_t> kernel_shape;
  std::vector<int64_t> dim_strides;
  std::vector<int64_t> padding;
  std::vector<int64_t> padding_from_back;
  for (int64_t i = 0; i < rank; i++) {
    kernel_shape.emplace_back(window_dims[i]);
    dim_strides.emplace_back(strides[i]);
    padding.emplace_back(paddings_front[i]);
    padding_from_back.emplace_back(paddings_back[i]);
  }

  padding.insert(padding.end(), padding_from_back.begin(),
                 padding_from_back.end());
  std::vector<popart::TensorId> result =
      g_comp->builder->aiOnnxOpset10().maxpool(
          {input->tensor_id}, 1, kernel_shape, 0, std::vector<int64_t>(),
          padding, 0, dim_strides, name);
  return new _odla_value(result[0],
                         {g_comp->builder->getTensorDataType(result[0]),
                          g_comp->builder->getTensorShape(result[0])},
                         name);
}

odla_value odla_PRelu(odla_value input, odla_float32 slope,
                      const odla_value_id value_id) {
  const auto& name =
      value_id ? std::string(reinterpret_cast<const char*>(value_id)) : "PRelu";

  std::vector<int64_t> input_shape =
      g_comp->builder->getTensorShape(input->tensor_id);
  int mul_total = 1;
  int rank = input_shape.size();
  for (int i = 0; i < rank; ++i) {
    mul_total *= input_shape[i];
  }
  std::vector<float> sloap_tmp(mul_total, slope);
  odla_value_shape dim_shape;
  dim_shape.size = rank;
  memcpy(dim_shape.dims, input_shape.data(), rank);
  auto sloap =
      odla_CreateConstant({ODLA_FLOAT32, dim_shape}, sloap_tmp.data(),
                          (const odla_value_id)(name + "_sloap").c_str());

  popart::TensorId result = g_comp->builder->aiOnnxOpset10().prelu(
      {input->tensor_id, sloap->tensor_id}, name);
  return new _odla_value(result,
                         {g_comp->builder->getTensorDataType(result),
                          g_comp->builder->getTensorShape(result)},
                         name);
}

odla_value odla_Selu(odla_value input, odla_float32 alpha, odla_float32 gamma,
                     const odla_value_id value_id) {
  const auto& name =
      value_id ? std::string(reinterpret_cast<const char*>(value_id)) : "Selu";

  popart::TensorId result = g_comp->builder->aiOnnxOpset10().selu(
      {input->tensor_id}, alpha, gamma, name);
  return new _odla_value(result,
                         {g_comp->builder->getTensorDataType(result),
                          g_comp->builder->getTensorShape(result)},
                         name);
}

odla_value odla_Sigmoid(odla_value input, const odla_value_id value_id) {
  const auto& name = value_id
                         ? std::string(reinterpret_cast<const char*>(value_id))
                         : "Sigmoid";

  popart::TensorId result =
      g_comp->builder->aiOnnxOpset10().sigmoid({input->tensor_id}, name);
  return new _odla_value(result,
                         {g_comp->builder->getTensorDataType(result),
                          g_comp->builder->getTensorShape(result)},
                         name);
}

odla_value odla_Tanh(odla_value input, const odla_value_id value_id) {
  const auto& name =
      value_id ? std::string(reinterpret_cast<const char*>(value_id)) : "Tanh";

  popart::TensorId result =
      g_comp->builder->aiOnnxOpset10().tanh({input->tensor_id}, name);
  return new _odla_value(result,
                         {g_comp->builder->getTensorDataType(result),
                          g_comp->builder->getTensorShape(result)},
                         name);
}

odla_value odla_TopK(odla_value input, odla_uint32 K, odla_bool largest,
                     odla_bool sorted, odla_uint32 axis,
                     odla_value_type output_value_type,
                     const odla_value_id value_id) {
  const auto& name =
      value_id ? std::string(reinterpret_cast<const char*>(value_id)) : "Topk";
  int64_t K_value[] = {K};
  auto K_tensor =
      odla_CreateConstant({ODLA_INT64, {.size = 1, .dims = {1}}}, K_value,
                          (const odla_value_id)(name + "_K").c_str());

  std::vector<popart::TensorId> results = g_comp->builder->aiOnnxOpset10().topk(
      {input->tensor_id, K_tensor->tensor_id}, axis, name);

  return new _odla_value(results[0],
                         {g_comp->builder->getTensorDataType(results[0]),
                          g_comp->builder->getTensorShape(results[0])},
                         name);
}

odla_values odla_PostProcess(odla_value orig_img_w, odla_value orig_img_h,
                             odla_value bb13, odla_value bb26, odla_value bb52,
                             const odla_value_id id) {
  const auto& name = id ? std::string(reinterpret_cast<const char*>(id)) : "";
  const static popart::OperatorIdentifier postprocess(
      popart::Domain::ai_graphcore, "PostProcess", 1, 5, 2);
  auto outs = g_comp->builder->customOp(
      postprocess, 1,
      {orig_img_w->tensor_id, orig_img_h->tensor_id, bb13->tensor_id,
       bb26->tensor_id, bb52->tensor_id},
      2, {});
  odla_value value_1 =
      new _odla_value(outs[0],
                      {g_comp->builder->getTensorDataType(outs[0]),
                       g_comp->builder->getTensorShape(outs[0])},
                      name + "0");
  odla_value value_2 =
      new _odla_value(outs[1],
                      {g_comp->builder->getTensorDataType(outs[1]),
                       g_comp->builder->getTensorShape(outs[1])},
                      name + "1");
  return odla_values{2, {value_1, value_2}};
}
