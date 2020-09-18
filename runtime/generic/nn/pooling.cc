//===- pooling.cc ---------------------------------------------------------===//
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

#include <limits>

#include "utils.h"

extern "C" {
/// pooling max helper func
/// general implementation for both nchw and nhwc format.
void _sn_rt_poolingmax_f32_helper(
    float* output, const float* data, int64_t batch, int64_t spatial_h,
    int64_t spatial_w, int64_t channel, int64_t output_h, int64_t output_w,
    int64_t kernel_h, int64_t kernel_w, int64_t stride_h, int64_t stride_w,
    int64_t pad_top, int64_t pad_bottom, int64_t pad_left, int64_t pad_right,
    bool is_nchw) {
  int64_t b, c, i, j, m, n;
  int64_t h_offset = -pad_top;
  int64_t w_offset = -pad_left;
  const float lattic = std::numeric_limits<float>::lowest();
  for (b = 0; b < batch; ++b) {
    for (c = 0; c < channel; ++c) {
      for (i = 0; i < output_h; ++i) {
        for (j = 0; j < output_w; ++j) {
          // auto o_index = c + (j + (i + b * output_h) * output_w) * channel;
          auto o_index = is_nchw ? _sn_rt_flatten_index_calculation(
                                       b, c, i, j, channel, output_h, output_w)
                                 : _sn_rt_flatten_index_calculation(
                                       b, i, j, c, output_h, output_w, channel);
          float max = std::numeric_limits<float>::lowest();
          for (m = 0; m < kernel_h; ++m) {
            for (n = 0; n < kernel_w; ++n) {
              auto curr_h = h_offset + i * stride_h + m;
              auto curr_w = w_offset + j * stride_w + n;
              // auto in_index =
              //    c + (curr_w + (curr_h + b * spatial_h) * spatial_w) *
              //    channel;
              auto in_index =
                  is_nchw
                      ? _sn_rt_flatten_index_calculation(
                            b, c, curr_h, curr_w, channel, spatial_h, spatial_w)
                      : _sn_rt_flatten_index_calculation(b, curr_h, curr_w, c,
                                                         spatial_h, spatial_w,
                                                         channel);
              bool valid = curr_h >= 0 && curr_h < spatial_h && curr_w >= 0 &&
                           curr_w < spatial_w;
              auto value = valid ? data[in_index] : lattic;
              max = max > value ? max : value;
            }
          }
          output[o_index] = max;
        }
      }
    }
  }
}

void _sn_rt_poolingmax_f32_nhwc(
    float* output, const float* data, int64_t batch, int64_t spatial_h,
    int64_t spatial_w, int64_t channel, int64_t output_h, int64_t output_w,
    int64_t kernel_h, int64_t kernel_w, int64_t stride_h, int64_t stride_w,
    int64_t pad_top, int64_t pad_bottom, int64_t pad_left, int64_t pad_right) {
  return _sn_rt_poolingmax_f32_helper(
      output, data, batch, spatial_h, spatial_w, channel, output_h, output_w,
      kernel_h, kernel_w, stride_h, stride_w, pad_top, pad_bottom, pad_left,
      pad_right, false /*is_nhwc*/);
}

void _sn_rt_poolingmax_f32_nchw(
    float* output, const float* data, int64_t batch, int64_t spatial_h,
    int64_t spatial_w, int64_t channel, int64_t output_h, int64_t output_w,
    int64_t kernel_h, int64_t kernel_w, int64_t stride_h, int64_t stride_w,
    int64_t pad_top, int64_t pad_bottom, int64_t pad_left, int64_t pad_right) {
  return _sn_rt_poolingmax_f32_helper(
      output, data, batch, spatial_h, spatial_w, channel, output_h, output_w,
      kernel_h, kernel_w, stride_h, stride_w, pad_top, pad_bottom, pad_left,
      pad_right, true /*is_nhwc*/);
}
}