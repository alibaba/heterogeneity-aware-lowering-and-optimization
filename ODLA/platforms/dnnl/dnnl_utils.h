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

#ifdef ODLA_BUILD_DNNL_GPU

#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/iterator>

#include <iomanip>
#include <iostream>

#include <CL/sycl.hpp>
using namespace sycl;
using namespace std;

#include "../examples/example_utils.hpp"

#endif

#include "dnnl.h"
#include "oneapi/dnnl/dnnl_sycl.hpp"

namespace dnnl_utils {

inline void floorbf_func(int len, int16_t *src, float *dst) { assert(0); }

inline void floorf_func(int len, float *src, float *dst) { assert(0); }

inline void rsqrtf_func(int len, float *src, float *dst) { assert(0); }

inline void rsqrtbf_func(int len, int16_t *src, float *dst) { assert(0); }

inline void topk_func(float *src, float *dst_data, int *dst_idx,
                      std::vector<int32_t> src_dims, uint32_t K, bool largest,
                      bool sorted, uint32_t axis) {
  assert(0);
}

inline void gather_func(char *params, int32_t *idx, size_t idx_size,
                        size_t inner_size, size_t outer_loop, size_t outer_size,
                        size_t byte_size, char *dst) {
  assert(0);
}

inline void erf_bf16_func(int16_t *src, float *dst, size_t len) { assert(0); }

inline void erf_func(float *src, float *dst, size_t len) { assert(0); }

inline void binary_s32_func(dnnl::algorithm alg, int32_t *lhs, int32_t *rhs,
                            int32_t *dst, int len) {
  assert(0);
}

void nms_func(float *boxes, float *scores, size_t batch_idx, size_t class_num,
              size_t num_boxes, size_t max_num_outputs, float score_threshold,
              float iou_threshold, int32_t *output_indices) {
  assert(0);
}

using SizeVector = std::vector<int>;
void topk_gpu(const dnnl::memory &src, const dnnl::memory &dst,
              const dnnl::memory &dst_idx_mem, SizeVector in_dims, int32_t axis,
              int before_num, int dim, int src_k, int count_vec,
              bool sort_value) {

  dnnl::engine eng = src.get_engine();

  size_t size = src.get_desc().get_size();
  size_t dst_size = dst.get_desc().get_size();

  auto sycl_queue = dnnl::sycl_interop::get_queue(dnnl::stream(eng));

  auto src_data = (float *)src.get_data_handle();
  auto dst_data = (float *)dst.get_data_handle();
  auto dst_idx = (float *)dst_idx_mem.get_data_handle();

  int n = size / sizeof(float);

  sycl::usm_allocator<float, sycl::usm::alloc::shared> alloc(sycl_queue);
  std::vector<float, decltype(alloc)> vals(n, alloc);
  std::vector<float, decltype(alloc)> keys(n, alloc);

  sycl_queue.memcpy(vals.data(), src_data, n * sizeof(float)).wait();
  // copy(policy, counting_begin, counting_begin + n, vals_begin);

  auto policy = oneapi::dpl::execution::make_device_policy(sycl_queue);
  auto counting_begin = oneapi::dpl::counting_iterator<int>{0};
  transform(policy, counting_begin, counting_begin + n, keys.begin(),
            [n](int i) { return i; });
  // 2. Sorting
  auto zipped_begin =
      oneapi::dpl::make_zip_iterator(vals.begin(), keys.begin());
  // stable sort by keys
  stable_sort(policy, zipped_begin, zipped_begin + n,
              [](auto lhs, auto rhs) { return get<0>(lhs) > get<0>(rhs); });

  std::copy(policy, std::begin(vals), std::begin(vals) + src_k, dst_data);
  std::copy(policy, std::begin(keys), std::begin(keys) + src_k, dst_idx);

  sycl_queue.wait();
  std::copy(dst_data, dst_data + src_k,
            std::ostream_iterator<float>{std::cout, " "});
  std::cout << std::endl;
  std::copy(dst_idx, dst_idx + src_k,
            std::ostream_iterator<int>{std::cout, " "});
  std::cout << std::endl;
}

inline int count(SizeVector dims, int start_ind, int end_ind) {
  size_t count = 1;
  for (size_t i = start_ind; i < end_ind; i++)
    count *= dims[i];
  return static_cast<int>(count);
}

inline int count(SizeVector dims, size_t start_ind = 0) {
  return count(dims, start_ind, dims.size());
}
void topk_func_gpu(const dnnl::memory &src, const dnnl::memory &dst,
                   const dnnl::memory &dst_idx, std::vector<int32_t> src_dims,
                   uint32_t K, bool largest, bool sorted, uint32_t axis) {

  std::cout << "topk_func_gpu "
            << " K" << K << "general code " << std::endl;
  auto in_dims = src_dims;
  size_t axis_dim;
  size_t axis_stride = 1;
  size_t axis_step = 1;
  int count_vec = 32;
  bool is_last_dim = false;
  int src_k = K;
  bool mode_max, sort_value;
  int dim, before_num;
  int axis_ = -1;
  if (axis_ < 0)
    axis_ += src_dims.size();
  axis = static_cast<size_t>(axis_);
  if (largest)
    mode_max = true;
  else
    mode_max = false;
  if (sorted)
    sort_value = true;
  else
    sort_value = false;
  int j;
  for (j = src_dims.size() - 1; j >= 0; j--) {
    if (src_dims[j] != 1)
      break;
  }
  if (static_cast<size_t>(j) == axis)
    is_last_dim = true;

  for (size_t i = 0; i < axis; i++) {
    axis_step *= src_dims[i];
  }
  axis_dim = src_dims[axis];
  for (size_t i = (axis + 1); i < src_dims.size(); i++) {
    axis_stride *= src_dims[i];
  }
  dim = static_cast<int>(src_dims[axis]);
  before_num = count(src_dims, 0, axis);

  topk_gpu(src, dst, dst_idx, in_dims, axis, before_num, dim, src_k, count_vec,
           sort_value);
}

} // namespace dnnl_utils
#endif
