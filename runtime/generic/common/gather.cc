//===- gather.cc ----------------------------------------------------------===//
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

extern "C" {
/// A dummy implementation.
void _sn_rt_gather_f32(float* out, const float* params, const int* indices,
                       int64_t param_col_size, int64_t indices_size) {
  const int elem_size = 4;
  const int copy_size = param_col_size * elem_size;
  for (int i = 0; i < indices_size; ++i) {
    for (int j = 0; j < param_col_size; ++j) {
      *out = params[indices[i] * param_col_size + j];
      out++;
    }
    /*const float* src = &params[indices[i] * param_col_size];
      std::memcpy(out, src, copy_size);
      out += copy_size;*/
  }
}
}