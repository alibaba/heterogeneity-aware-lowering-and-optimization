//===- broadcast.h --------------------------------------------------------===//
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

#ifndef HALO_LIB_RUNTIME_GENERIC_MATH_BROADCAST_H_
#define HALO_LIB_RUNTIME_GENERIC_MATH_BROADCAST_H_

#include <iostream>
#include <numeric>

extern "C" {
void _sn_rt_broadcast_strides_calculation(
    int64_t* lhs_strides, int64_t* rhs_strides, const int64_t* result_shape,
    const int64_t* lhs_shape, const int64_t* rhs_shape, int32_t dims) {
  lhs_strides[dims - 1] = 1;
  rhs_strides[dims - 1] = 1;
  for (int i = dims - 2; i >= 0; i--) {
    lhs_strides[i] = lhs_strides[i + 1] * lhs_shape[i + 1];
    rhs_strides[i] = rhs_strides[i + 1] * rhs_shape[i + 1];
  }
  for (int i = 0; i < dims; ++i) {
    if (lhs_shape[i] != result_shape[i]) {
      lhs_strides[i] = 0;
    }
    if (rhs_shape[i] != result_shape[i]) {
      rhs_strides[i] = 0;
    }
  }
}
}

#endif // HALO_LIB_RUNTIME_GENERIC_MATH_BROADCAST_H_