//===- odla_dnnl_quantization.cc ------------------------------------------===//
//
// Copyright (C) 2019-2021 Alibaba Group Holding Limited.
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

#include "ODLA/odla_common.h"
#include "ODLA/ops/odla_ops_quantization.h"
#include "odla_dnnl.h"

static odla_value_shape ExpandShape(const odla_value_shape& input_shape,
                                    int axis) {
  auto shape = input_shape;
  for (int i = 0; i < shape.size; ++i) {
    if (i != axis) {
      shape.dims[i] = 1;
    }
  }
  return shape;
}

odla_value odla_Dequantize(odla_value input, odla_value scale,
                           odla_value zero_point, odla_int32 axis,
                           odla_element_type target_data_type,
                           const odla_value_id value_id) {
  // TODO: If scale & zero_point are constants, we can use reorder + append_sum
  // post op.
  assert(target_data_type == ODLA_FLOAT32);
  auto ret_md = getMemoryDesc(input->shape, target_data_type);
  auto ret_mem = dnnl::memory(ret_md, g_comp->eng);
  if (zero_point != nullptr) {
    zero_point = odla_Cast(zero_point, ODLA_FLOAT32, nullptr);
  }
  if (scale->shape.size > 0) {
    const auto& shape = ExpandShape(input->shape, axis);
    scale = odla_Reshape(scale, shape, nullptr);
    if (zero_point != nullptr) {
      zero_point = odla_Reshape(zero_point, shape, nullptr);
    }
  }
  input = odla_Cast(input, ODLA_FLOAT32, nullptr);
  if (zero_point != nullptr) {
    input = odla_Sub(input, zero_point, nullptr);
  }
  auto v = odla_Mul(odla_Cast(input, ODLA_FLOAT32, nullptr), scale, value_id);
  InterpretIfNeeded();
  return v;
}

odla_value odla_Quantize(odla_value input, odla_value scale,
                         odla_value zero_point, odla_int32 axis,
                         odla_element_type target_data_type,
                         const odla_value_id value_id) {
  // TODO: If scale & zero_point are constants, we can use reorder + scale post
  // op.

  assert(target_data_type == ODLA_INT8 || target_data_type == ODLA_UINT8);
  auto ret_md = getMemoryDesc(input->shape, target_data_type);
  auto ret_mem = dnnl::memory(ret_md, g_comp->eng);
  bool needs_expand = GetTotalElements(scale->shape) > 1 &&
                      axis != input->shape.dims[input->shape.size - 1];
  const auto& shape = ExpandShape(input->shape, axis);
  if (needs_expand) {
    scale = odla_Reshape(scale, shape, nullptr);
  }
  input = odla_Round(odla_Div(input, scale, nullptr), nullptr);
  if (zero_point != nullptr) {
    zero_point = odla_Cast(zero_point, input->elem_type, nullptr);
    if (needs_expand) {
      zero_point = odla_Reshape(zero_point, shape, nullptr);
    }
    input = odla_Add(input, zero_point, nullptr);
  }

  auto v = odla_Cast(input, target_data_type, value_id);

  InterpretIfNeeded();
  v->elem_type = target_data_type;
  return v;
}