//===- onehot.cc ----------------------------------------------------------===//
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
void _sn_rt_onehot_f32(float* out, const int* indices, const int* depth,
                       const float* off_value, const float* on_value,
                       int64_t noe_indices) {
  for (int i = 0; i < noe_indices; ++i) {
    for (int d = 0; d < *depth; ++d) {
      *out = (d == indices[i]) ? *on_value : *off_value;
      out++;
    }
  }
}
}
