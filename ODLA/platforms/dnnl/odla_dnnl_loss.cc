//===- odla_dnnl_rnn.cc ---------------------------------------------------===//
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

#include <cstring>

#include "ODLA/odla_common.h"
#include "ODLA/ops/odla_ops_nn.h"
#include "odla_dnnl.h"

template <typename T>
static void nlll(int64_t batch, int64_t cls, int64_t s, const T* data,
                 const int64_t* gt, int ignored, odla_reduction_mode reduction,
                 const float* weight, float* dst) {
  float denom = 0;
  for (int64_t b = 0; b < batch; ++b) {
    for (int64_t k = 0; k < s; ++k) {
      auto c = gt[b * s + k];
      if (c == ignored || c < 0 || c >= cls) {
        continue;
      }
      float v = -data[b * s * cls + c * s + k];
      if (weight != nullptr) {
        v *= weight[c];
        denom += weight[c];
      } else {
        denom += 1;
      }
      if (reduction != ODLA_REDUCE_NONE) {
        *dst += v;
      } else {
        dst[b * s + k] = v;
      }
    }
  }
  if (reduction == ODLA_REDUCE_MEAN) {
    *dst /= denom;
  }
}
odla_value odla_NegativeLogLikeliHoodLoss(odla_value input, odla_value gt,
                                          odla_int32 ignored,
                                          odla_reduction_mode reduction,
                                          odla_value weight,
                                          odla_value_shape output_shape,
                                          odla_value_id value_id) {
  auto rank = input->shape.size;
  assert(rank >= 2);
  auto batch = input->shape.dims[0];
  auto cls = input->shape.dims[1];
  auto seq = GetTotalElements(input->shape) / batch / cls;
  auto ret_md = getMemoryDesc(output_shape, ODLA_FLOAT32);
  auto ret_mem = dnnl::memory(ret_md, g_comp->eng);
  float* dst = static_cast<float*>(ret_mem.get_data_handle());
  std::fill_n(dst, GetTotalElements(output_shape), .0F);
  auto op = [batch, cls, seq, input, gt, ignored, reduction, weight, dst]() {
    const float* data = static_cast<const float*>(input->mem.get_data_handle());
    const float* w =
        weight != nullptr
            ? static_cast<const float*>(weight->mem.get_data_handle())
            : nullptr;
    const int64_t* label =
        static_cast<const int64_t*>(gt->mem.get_data_handle());
    nlll(batch, cls, seq, data, label, ignored, reduction, w, dst);
  };
  add_op(op);
  InterpretIfNeeded();

  // if (weight != nullptr) {
  //  odla_value t = CreateValue(ret_mem, output_shape, nullptr);
  //  return odla_Mul(t, weight, value_id);
  // }
  odla_value v = CreateValue(ret_mem, output_shape, value_id);
  return v;
}
