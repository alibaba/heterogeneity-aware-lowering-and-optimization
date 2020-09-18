//===- mul.cc -------------------------------------------------------------===//
//
// Copyright (C) 2019-2020 Alibaba Group Holding Limited.
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

#include <stdint.h>

#include "broadcast.h"

extern "C" {
/// A dummy implementation.
void _sn_rt_mul_f32(float* out, const float* lhs, const float* rhs,
                    int64_t ret_size, bool need_broadcast,
                    const int64_t* ret_shape, const int64_t* lhs_shape,
                    const int64_t* rhs_shape, int32_t dims) {
  if (!need_broadcast) {
    for (int64_t i = 0; i < ret_size; ++i) {
      out[i] = lhs[i] * rhs[i];
    }
    return;
  }
  int64_t lhs_strides[dims];
  int64_t rhs_strides[dims];
  _sn_rt_broadcast_strides_calculation(lhs_strides, rhs_strides, ret_shape,
                                       lhs_shape, rhs_shape, dims);
  int pos[dims];
  std::fill_n(&pos[0], dims, 0);
  for (size_t i = 0; i < ret_size; ++i) {
    size_t lhs_index =
        std::inner_product(&pos[0], pos + dims, &lhs_strides[0], 0UL);
    size_t rhs_index =
        std::inner_product(&pos[0], pos + dims, &rhs_strides[0], 0UL);
    *out++ = lhs[lhs_index] * rhs[rhs_index];
    int c = 1;
    for (int i = dims - 1; i >= 0 && c == 1; --i) {
      pos[i] += c;
      if (pos[i] >= ret_shape[i]) {
        pos[i] = 0;
      } else {
        c = 0;
      }
    }
  }
}
}