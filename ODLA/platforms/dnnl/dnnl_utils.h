//===- dnnl_utils.h -------------------------------------------------------===//
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

#include <immintrin.h>

#if defined(__AVX512F__)
#include "dnnl_utils_avx512.h"
#else

namespace dnnl_utils {

inline void floorbf_func(int len, float* src, float* dst) { assert(0); }

inline void floorf_func(int len, float* src, float* dst) { assert(0); }

inline void rsqrtf_func(int len, float* src, float* dst) { assert(0); }

inline void rsqrtbf_func(int len, float* src, float* dst) { assert(0); }

inline void cast_fp32int8_func(int len, float* src, int8_t* dst) { assert(0); }
inline void cast_int32fp32_func(int len, int* src, float* dst) { assert(0); }
inline void cast_fp32int32_func(int len, float* src, int* dst) { assert(0); }
inline void cast_int8fp32_func(int len, int8_t* src, float* dst) { assert(0); }
inline void topk_func(float* src, float* dst_data, int* dst_idx,
                      std::vector<int32_t> src_dims, uint32_t K, bool largest,
                      bool sorted, uint32_t axis) {
  assert(0);
}

inline void gather_byte1_func(int8_t* params, int32_t* idx, size_t batch_size,
                              size_t idx_size, size_t inner_size, int8_t* dst) {
  assert(0);
}

inline void gather_byte2_func(int16_t* params, int32_t* idx, size_t batch_size,
                              size_t idx_size, size_t inner_size,
                              int16_t* dst) {
  assert(0);
}

inline void gather_byte4_func(float* params, int32_t* idx, size_t batch_size,
                              size_t idx_size, size_t inner_size, float* dst) {
  assert(0);
}

inline void erf_p(float* src, float* dst, size_t len) { assert(0); }

inline void binary_s32_func(dnnl::algorithm alg, int32_t* lhs, int32_t* rhs,
                            int32_t* dst, int len) {
  assert(0);
}

void nms_func(float* boxes, float* scores, size_t batch_idx, size_t class_num,
              size_t num_boxes, size_t max_num_outputs, float score_threshold,
              float iou_threshold, int32_t* output_indices) {
  assert(0);
}

} // namespace dnnl_utils
#endif
