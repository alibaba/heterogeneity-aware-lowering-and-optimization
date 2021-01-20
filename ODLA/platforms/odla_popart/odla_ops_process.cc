//===- odla_ops_process.cc ------------------------------------------------===//
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
#include <iostream>
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
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "common.h"
#include "odla_popart.h"

#if !defined(ODLA_VERSION_NUMBER) || (ODLA_VERSION_NUMBER < 50)
#error This library requires minimum ODLA version 0.5
#endif

extern thread_local odla_computation g_comp;

odla_value odla_Cast(odla_value input, odla_element_type target_type,
                     const odla_value_id value_id) {
  const auto& name =
      value_id ? std::string(reinterpret_cast<const char*>(value_id)) : "Cast";
  std::string type_name = GetTypeName(target_type);
  popart::TensorId result =
      g_comp->builder->aiOnnxOpset10().cast({input->tensor_id}, type_name);

  return new _odla_value(result,
                         {g_comp->builder->getTensorDataType(result),
                          g_comp->builder->getTensorShape(result)},
                         name);
}

odla_value odla_Concat(odla_values inputs, odla_int32 axis,
                       odla_value_shape output_shape,
                       const odla_value_id value_id) {
  const auto& name = value_id
                         ? std::string(reinterpret_cast<const char*>(value_id))
                         : "Concat";
  std::vector<popart::TensorId> onnx_inputs;
  for (int i = 0; i < inputs.size; ++i) {
    onnx_inputs.emplace_back(inputs.values[i]->tensor_id);
  }
  popart::TensorId result =
      g_comp->builder->aiOnnxOpset10().concat(onnx_inputs, axis);

  return new _odla_value(result,
                         {g_comp->builder->getTensorDataType(result),
                          g_comp->builder->getTensorShape(result)},
                         name);
}

odla_value odla_ExpandDims(odla_value input, odla_value_shape output_dims,
                           const odla_value_id value_id) {
  const auto& name = value_id
                         ? std::string(reinterpret_cast<const char*>(value_id))
                         : "Expand";
  std::vector<int64_t> shape_data;
  for (int i = 0; i < output_dims.size; ++i) {
    shape_data.emplace_back(output_dims.dims[i]);
  }
  auto output_shape = odla_CreateConstant(
      {ODLA_INT64, {.size = 1, .dims = {output_dims.size}}}, shape_data.data(),
      (const odla_value_id) "output_shape");
  popart::TensorId result = g_comp->builder->aiOnnxOpset10().expand(
      {input->tensor_id, output_shape->tensor_id});

  return new _odla_value(result,
                         {g_comp->builder->getTensorDataType(result),
                          g_comp->builder->getTensorShape(result)},
                         name);
}

odla_value odla_Pad(odla_value input, const odla_uint32* padding_front,
                    const odla_uint32* padding_back,
                    odla_value_shape output_dims,
                    const odla_value_id value_id) {
  const auto& name =
      value_id ? std::string(reinterpret_cast<const char*>(value_id)) : "Pad";

  std::vector<int64_t> padding;
  std::vector<int64_t> padding_from_back;
  int rank = input->tensor_info.rank();
  for (int64_t i = 0; i < rank; i++) {
    padding.emplace_back(padding_front[i]);
    padding_from_back.emplace_back(padding_back[i]);
  }

  padding.insert(padding.end(), padding_from_back.begin(),
                 padding_from_back.end());

  popart::TensorId result = g_comp->builder->aiOnnxOpset10().pad(
      {input->tensor_id}, padding, "constant", 0.0f);

  return new _odla_value(result,
                         {g_comp->builder->getTensorDataType(result),
                          g_comp->builder->getTensorShape(result)},
                         name);
}

odla_value odla_Resize(odla_value input, odla_interpolation_mode interpolation,
                       odla_resize_coordinate_mode mode, odla_uint32 axes_mask,
                       odla_value_shape output_dims,
                       const odla_value_id value_id) {
  const auto& name = value_id
                         ? std::string(reinterpret_cast<const char*>(value_id))
                         : "Resize";
  std::string interpolation_mode_name =
      GetResizeInterpolationModeName(interpolation);
  std::vector<float> shape_data;
  auto input_shape = g_comp->builder->getTensorShape(input->tensor_id);
  for (int i = 0; i < output_dims.size; ++i) {
    shape_data.emplace_back(output_dims.dims[i] / (float)input_shape[i]);
  }
  auto scales = odla_CreateConstant(
      {ODLA_FLOAT32, {.size = 1, .dims = {output_dims.size}}},
      shape_data.data(), (const odla_value_id) "Scale_shape");
  popart::TensorId result = g_comp->builder->aiOnnxOpset10().resize(
      {input->tensor_id, scales->tensor_id}, interpolation_mode_name);

  return new _odla_value(result,
                         {g_comp->builder->getTensorDataType(result),
                          g_comp->builder->getTensorShape(result)},
                         name);
}

odla_value odla_Shape(odla_value input, odla_value_shape output_dims,
                      const odla_value_id value_id) {
  const auto& name =
      value_id ? std::string(reinterpret_cast<const char*>(value_id)) : "Shape";

  popart::TensorId result =
      g_comp->builder->aiOnnxOpset10().shape({input->tensor_id});

  return new _odla_value(result,
                         {g_comp->builder->getTensorDataType(result),
                          g_comp->builder->getTensorShape(result)},
                         name);
}

odla_value odla_Squeeze(odla_value input, odla_size_t num_of_axes,
                        const odla_uint32* axes, odla_value_shape output_dims,
                        const odla_value_id value_id) {
  const auto& name = value_id
                         ? std::string(reinterpret_cast<const char*>(value_id))
                         : "Squeeze";
  std::vector<int64_t> axes_to_squeeze;
  for (int i = 0; i < num_of_axes; ++i) {
    axes_to_squeeze.emplace_back(axes[i]);
  }
  popart::TensorId result = g_comp->builder->aiOnnxOpset10().squeeze(
      {input->tensor_id}, axes_to_squeeze);

  return new _odla_value(result,
                         {g_comp->builder->getTensorDataType(result),
                          g_comp->builder->getTensorShape(result)},
                         name);
}
