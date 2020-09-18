//===- conv.cc ------------------------------------------------------------===//
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

extern "C" {
/// conv2d in nhwc.

void _sn_rt_conv2d_f32_nhwc(
    float* output, const float* data, const float* kernel, int64_t batch,
    int64_t spatial_h, int64_t spatial_w, int64_t channel, int64_t output_h,
    int64_t output_w, int64_t output_channel, int64_t kernel_h,
    int64_t kernel_w, int64_t stride_h, int64_t stride_w, int64_t pad_top,
    int64_t pad_bottom, int64_t pad_left, int64_t pad_right);
}