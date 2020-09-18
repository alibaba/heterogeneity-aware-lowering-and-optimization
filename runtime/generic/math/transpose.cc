//===- transpose.cc -------------------------------------------------------===//
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

#include <cstring>
#include <iostream>
#include <numeric>

extern "C" {
/// A dummy implementation.
void _sn_rt_transpose_f32(float* out, const float* data, const int* perm,
                          const int64_t* orig_shape, int dims) {
  int orig_strides[dims];
  std::fill_n(&orig_strides[0], dims, 1);
  for (int i = dims - 2; i >= 0; --i) {
    orig_strides[i] = orig_strides[i + 1] * orig_shape[i + 1];
  }
  int64_t new_shape[dims];
  int perm_strides[dims];
  size_t elem_cnt = 1;
  for (int i = 0; i < dims; ++i) {
    new_shape[i] = orig_shape[perm[i]];
    perm_strides[i] = orig_strides[perm[i]];
    elem_cnt *= orig_shape[i];
  }
  int pos[dims];
  std::fill_n(&pos[0], dims, 0);
  const size_t elem_size = 4;
  char* buf = reinterpret_cast<char*>(out); // NOLINT
  for (size_t i = 0; i < elem_cnt; ++i) {
    size_t offset =
        std::inner_product(&pos[0], pos + dims, &perm_strides[0], 0UL);
    const char* src =
        reinterpret_cast<const char*>(data) + offset * elem_size; // NOLINT
    std::memcpy(&buf[i * elem_size], src, elem_size);
    int c = 1;
    for (int i = dims - 1; i >= 0 && c == 1; --i) {
      pos[i] += c;
      if (pos[i] >= new_shape[i]) {
        pos[i] = 0;
      } else {
        c = 0;
      }
    }
  }
}
}