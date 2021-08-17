//===- reduce.cc ----------------------------------------------------------===//
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
#include <cstdint>
#include <cstring>

extern "C" {
/// Helper function
static void _sn_rt_reduce_helper(float* data, int64_t* in_dims,
                                 int in_dims_size, const int64_t* axies,
                                 int axies_size, int index) {
  if (index == axies_size) {
    return;
  }
  int64_t curr = axies[index];
  int64_t dim = in_dims[curr];
  int64_t before, after;
  before = after = 1;
  for (int i = 0; i < in_dims_size; ++i) {
    if (i < curr) {
      before *= in_dims[i];
    } else if (i > curr) {
      after *= in_dims[i];
    } else {
      in_dims[i] = 1;
    }
  }
  for (int i = 0; i < before; ++i) {
    for (int j = 0; j < dim; ++j) {
      for (int k = 0; k < after; ++k) {
        auto o_index = k + i * after;
        auto i_index = k + (j + i * dim) * after;
        if (j == 0) {
          data[o_index] = data[i_index];
        } else {
          data[o_index] += data[i_index];
        }
      }
    }
  }
  for (int i = 0; i < before * after; ++i) {
    data[i] /= (float)dim;
  }
  _sn_rt_reduce_helper(data, in_dims, in_dims_size, axies, axies_size,
                       index + 1);
}

/// reducemean with fp32.
void _sn_rt_reduce_mean_f32(float* result, const float* data,
                            const int64_t* shape, const int32_t* axis,
                            int64_t num_of_elements, int32_t dim,
                            int32_t axis_dim) {
  constexpr int DIM_BEFORE = 4;
  int64_t in_dims[] = {1, 1, 1, 1};
  constexpr int in_dims_size = sizeof(in_dims) / sizeof(in_dims[0]);
  int64_t axies[axis_dim];
  const int axies_size = axis_dim;
  for (int i = 0; i < axis_dim; ++i) {
    axies[i] = 1;
  }
  const int DIM_EXPAND = DIM_BEFORE - dim;
  for (auto i = 0; i != DIM_BEFORE; ++i) {
    in_dims[i] = i >= DIM_EXPAND ? shape[i - DIM_EXPAND] : 1;
  }
  for (auto i = 0; i != axis_dim; ++i) {
    axies[i] = axis[i] + DIM_EXPAND;
  }
  // std::sort(axies.begin(), axies.end());
  // axies.erase(std::unique(axies.begin(), axies.end()), axies.end());
  int64_t before, after;
  before = after = 1;
  int64_t curr = axies[0];
  int64_t reduced_dim = in_dims[curr];
  for (int i = 0; i < in_dims_size; ++i) {
    if (i < curr) {
      before *= in_dims[i];
    } else if (i > curr) {
      after *= in_dims[i];
    } else {
      in_dims[i] = 1;
    }
  }
  float temp[before * after];
  for (int i = 0; i < before; ++i) {
    for (int j = 0; j < reduced_dim; ++j) {
      for (int k = 0; k < after; ++k) {
        auto o_index = k + i * after;
        auto i_index = k + (j + i * reduced_dim) * after;
        if (j == 0) {
          temp[o_index] = data[i_index];
        } else {
          temp[o_index] += data[i_index];
        }
      }
    }
  }
  for (int i = 0; i < before * after; ++i) {
    temp[i] /= (float)reduced_dim;
  }

  if (axies_size > 1) {
    _sn_rt_reduce_helper(temp, in_dims, in_dims_size, axies, axies_size, 1);
  }

  int64_t result_noe = 1;
  for (auto i : in_dims) {
    result_noe *= i;
  }
  for (int i = 0; i < result_noe; ++i) {
    result[i] = temp[i];
  }
}

void _sn_rt_argmax_f32(int64_t* result, const float* data, const int64_t* shape,
                       const int32_t* axis, int64_t num_of_elements,
                       int32_t dim, int32_t axis_dim) {
  // axis is a scalar for argmax/argmin
  int axis_adj = axis[0];
  if (axis[0] < 0) {
    axis_adj += dim;
  }
  int64_t before, after;
  before = after = 1;
  for (int i = 0; i < dim; ++i) {
    if (i < axis_adj) {
      before *= shape[i];
    } else if (i > axis_adj) {
      after *= shape[i];
    }
  }
  int64_t reduced_dims = shape[axis_adj];
  for (int i = 0; i < before; ++i) {
    float value_max[after];
    for (int j = 0; j < reduced_dims; ++j) {
      for (int k = 0; k < after; ++k) {
        auto o_index = k + i * after;
        auto i_index = k + (j + i * reduced_dims) * after;
        if (j == 0) {
          value_max[k] = data[i_index];
          result[o_index] = 0;
        } else if (value_max[k] < data[i_index]) {
          value_max[k] = data[i_index];
          result[o_index] = j;
        }
      }
    }
  }
}
}