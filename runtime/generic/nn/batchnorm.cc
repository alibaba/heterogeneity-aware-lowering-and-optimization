//===- batchnorm.cc -------------------------------------------------------===//
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

#include <cmath>
#include <cstdint>
#include <limits>

#include "utils.h"

extern "C" {

static void _sn_rt_bn_f32_helper(
    float* output, const float* data, const float* mean, const float* variance,
    const float* offset, const float* scale, int64_t batch, int64_t spatial_h,
    int64_t spatial_w, int64_t channel, bool scalar_offset, bool scalar_scale,
    float epsilon, float prescaling, bool is_nchw) {
  int b, c, i, j;
  for (c = 0; c < channel; ++c) {
    float mean_v = mean[c];
    float variance_v = variance[c];
    float one_over_sqrt_variance = 1.0 / std::sqrt(variance_v + epsilon);
    float offset_v = scalar_offset ? *offset : offset[c];
    float scale_v = scalar_offset ? *scale : scale[c];
    float const_part = -scale_v * mean_v * one_over_sqrt_variance + offset_v;
    for (b = 0; b < batch; ++b) {
      for (i = 0; i < spatial_h; ++i) {
        for (j = 0; j < spatial_w; ++j) {
          auto index = is_nchw ? _sn_rt_flatten_index_calculation(
                                     b, c, i, j, channel, spatial_h, spatial_w)
                               : _sn_rt_flatten_index_calculation(
                                     b, i, j, c, spatial_h, spatial_w, channel);
          output[index] =
              scale_v * data[index] * one_over_sqrt_variance + const_part;
        }
      }
    }
  }
}

/// batch normalization in nhwc.
void _sn_rt_bn_f32_nhwc(float* output, const float* data, const float* mean,
                        const float* variance, const float* offset,
                        const float* scale, int64_t batch, int64_t spatial_h,
                        int64_t spatial_w, int64_t channel, bool scalar_offset,
                        bool scalar_scale, float epsilon, float prescaling) {
  _sn_rt_bn_f32_helper(output, data, mean, variance, offset, scale, batch,
                       spatial_h, spatial_w, channel, scalar_offset,
                       scalar_scale, epsilon, prescaling, false /*is_nchw*/);
}

/// batch normalization in nchw.
void _sn_rt_bn_f32_nchw(float* output, const float* data, const float* mean,
                        const float* variance, const float* offset,
                        const float* scale, int64_t batch, int64_t spatial_h,
                        int64_t spatial_w, int64_t channel, bool scalar_offset,
                        bool scalar_scale, float epsilon, float prescaling) {
  _sn_rt_bn_f32_helper(output, data, mean, variance, offset, scale, batch,
                       spatial_h, spatial_w, channel, scalar_offset,
                       scalar_scale, epsilon, prescaling, true /*is_nchw*/);
}
}