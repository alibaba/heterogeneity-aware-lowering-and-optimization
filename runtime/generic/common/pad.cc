//===- pad.cc -------------------------------------------------------------===//
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
/// Padding HW dimensions for NHWC layout.
static void _sn_rt_pad_4_nhwc(float* __restrict__ dst,
                              const float* __restrict__ src, int batch,
                              int orig_h, int orig_w, int channels, int pad_top,
                              int pad_bottom, int pad_left, int pad_right) {
  const int elem_size = 4;
  const int new_h = orig_h + pad_top + pad_bottom;
  const int new_w = orig_w + pad_left + pad_right;
  const int copy_len = orig_w * channels;
  const int dst_top_skip = pad_top * new_w * channels;
  const int dst_bottom_skip = pad_bottom * new_w * channels;
  const int dst_left_skip = pad_left * channels;
  const int dst_right_skip = pad_right * channels;

  for (int n = 0; n < batch; ++n) {
    std::memset(dst, 0, dst_top_skip * elem_size);
    dst += dst_top_skip;
    for (int r = 0; r < orig_h; ++r) {
      std::memset(dst, 0, dst_left_skip * elem_size);
      dst += dst_left_skip;
      std::memcpy(dst, src, copy_len * elem_size);
      src += copy_len;
      dst += copy_len;
      std::memset(dst, 0, dst_right_skip * elem_size);
      dst += dst_right_skip;
    }
    std::memset(dst, 0, dst_bottom_skip * elem_size);
    dst += dst_bottom_skip;
  }
}

void _sn_rt_pad_f32(float* __restrict__ dst, const float* __restrict__ src,
                    int dims, const int64_t* orig_shape,
                    const int32_t* paddings) {
  if (dims == 4 && paddings[0] == 0 && paddings[1] == 0 && paddings[6] == 0 &&
      paddings[7] == 0) {
    _sn_rt_pad_4_nhwc(dst, src, orig_shape[0], orig_shape[1], orig_shape[2],
                      orig_shape[3], paddings[2], paddings[3], paddings[4],
                      paddings[5]);
  } else {
    // std::cout << "unsupported padding)";
  }
}
} // end of extern "C"