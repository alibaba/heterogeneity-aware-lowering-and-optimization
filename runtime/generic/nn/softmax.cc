//===- softmax.cc ---------------------------------------------------------===//
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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

extern "C" {
/// softmax with fp32.
void _sn_rt_softmax_f32(float* result, const float* logits,
                        const int64_t* shape, int32_t axis, int32_t dim,
                        int64_t num_of_elements) {
  int64_t after = 1;
  for (int d = axis; d < dim; ++d) {
    after *= shape[d];
  }
  float temp[after];
  // std::vector<float> temp(after);
  int64_t before = num_of_elements / after;
  for (int i = 0; i < before; ++i) {
    float sum = 0;
    int64_t index = i * after;

    float max_value = *std::max_element(logits + index, logits + index + after);
    for (int j = 0; j < after; ++j) {
      temp[j] = std::exp(logits[j + index] - max_value);
      sum += temp[j];
    }
    for (int j = 0; j < after; ++j) {
      result[j + index] = temp[j] / sum;
    }
  }
}
}