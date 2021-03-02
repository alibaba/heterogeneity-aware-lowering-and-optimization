//===- dnnl_utils_avx512.h ------------------------------------------------===//
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
#include <omp.h>

namespace dnnl_utils {

static int calculat_offset(int len, int vec_size) {
  /*
  calculate the offset when using intrinsics.
  example:
    when len is 108 vec_size is 32 when using bf16
    the result is 108 % 32 = 12
    so we need to set the mask to 0b00000000000000000000111111111111
  */
  int offset = len;
  int expo = 0;
  int dst = 0;
  while (offset - vec_size > 0) {
    offset -= vec_size;
  }
  while (offset > 0) {
    dst += pow(2, expo);
    offset -= 1;
    expo += 1;
  }
  return dst;
}

#if defined(__GNUC__) && (__GNUC__ > 9)
inline void binary_s32_func(dnnl::algorithm alg, int32_t* lhs, int32_t* rhs,
                            int32_t* dst, int len) {
  int i = 0;
  int vec_size = 512 / 32;
  __mmask16 mask16 = 0xFFFF;
  __m512i (*__mm512_binary_op)(__m512i, __m512i);
  switch (alg) {
    case dnnl::algorithm::binary_add:
      __mm512_binary_op = [](__m512i a, __m512i b) {
        return _mm512_add_epi32(a, b);
      };
      break;
    case dnnl::algorithm::binary_mul:
      __mm512_binary_op = [](__m512i a, __m512i b) {
        return _mm512_mul_epi32(a, b);
      };
      break;
    default:
      break;
  }
  for (; i <= len - vec_size; i += vec_size) {
    auto a1 = _mm512_loadu_epi32(lhs + i);
    auto b1 = _mm512_loadu_epi32(rhs + i);
    auto out1 = __mm512_binary_op(a1, b1);
    _mm512_mask_storeu_epi32(dst + i, mask16, out1);
  }
  if (len - i) {
    auto tail_mask = calculat_offset(len - i, vec_size);
    auto a1 = _mm512_maskz_loadu_epi32(tail_mask, lhs + i);
    auto b1 = _mm512_maskz_loadu_epi32(tail_mask, rhs + i);
    auto out1 = __mm512_binary_op(a1, b1);
    _mm512_mask_storeu_epi32(dst + i, tail_mask, out1);
  }
}
#else
inline void binary_s32_func(dnnl::algorithm alg, int32_t* lhs, int32_t* rhs,
                            int32_t* dst, int len) {
  assert(0);
}
#endif

inline __m512 _mm512_cvtbf16f32_load(__mmask16 mask, void* mem_addr) {
  auto dst = _mm512_slli_epi32(
      _mm512_cvtepu16_epi32(_mm256_maskz_loadu_epi16(mask, mem_addr)), 0x10);
  return _mm512_castsi512_ps(dst);
}

inline void gather_func(char* params, int32_t* idx, size_t batch_size,
                        size_t idx_size, size_t axis_size, size_t inner_size,
                        size_t byte_size, char* dst) {
  size_t slice_bytes = inner_size * byte_size;
#pragma omp parallel for
  for (int i = 0; i < batch_size; i++) {
    for (int j = 0; j < idx_size; j++) {
      int32_t curr_idx = idx[j];
      memcpy(dst + (i * idx_size + j) * slice_bytes,
             params + (i * axis_size + curr_idx) * slice_bytes, slice_bytes);
    }
  }
}

#if defined(__GNUC__) && (__GNUC__ > 9)
inline void floorbf_func(int len, int16_t* src, float* dst) {
  int i = 0;
  int vec_size = 512 / 16;
  __mmask16 mask16 = 0xFFFF;
  auto alpha_vec = _mm512_set1_ps(0.0);
  for (; i <= len - vec_size; i += vec_size) {
    auto a0 = _mm512_cvtbf16f32_load(mask16, src + i);
    auto a1 = _mm512_cvtbf16f32_load(mask16, src + i + 16);

    auto out0 = _mm512_floor_ps(a0);
    auto out1 = _mm512_floor_ps(a1);

    auto C_bf16 = _mm512_cvtne2ps_pbh(out1, out0);
    _mm512_mask_storeu_ps(dst + i / 2, mask16, _mm512_castsi512_ps(C_bf16));
  }
  if ((len - i) > 16) {
    auto a0 = _mm512_cvtbf16f32_load(mask16, src + i);
    auto out0 = _mm512_floor_ps(a0);
    auto C_bf16 = _mm512_cvtneps_pbh(out0);
    _mm256_storeu_ps(dst + i / 2, _mm256_castsi256_ps(C_bf16));
    i += vec_size / 2;
  }
  if (len - i) {
    __mmask16 tail_mask = calculat_offset(i, vec_size);
    auto a0 = _mm512_cvtbf16f32_load(tail_mask, src + i);
    auto out0 = _mm512_floor_ps(a0);
    auto C_bf16 = _mm512_cvtneps_pbh(out0);
    _mm256_mask_storeu_ps(dst + i, tail_mask, _mm256_castsi256_ps(C_bf16));
  }
}
#elif defined(__GNUC__) && (__GNUC__ > 8)
inline void floorbf_func(int len, int16_t* src, float* dst) {
  int i = 0;
  int vec_size = 512 / 32;
  __mmask16 mask16 = 0xFFFF;
  auto alpha_vec = _mm512_set1_ps(0.0);
  auto tail_mask = calculat_offset(len, vec_size);
  for (; i <= len - vec_size; i += vec_size) {
    auto a0 = _mm512_cvtbf16f32_load(mask16, src + i);
    auto out0 = _mm512_floor_ps(a0);
    auto C_bf16 = _mm512_cvtneps_pbh(out0);
    _mm256_storeu_ps(dst + i, _mm256_castsi256_ps(C_bf16));
  }
  if (len - i) {
    auto a0 = _mm512_cvtbf16f32_load(tail_mask, src + i);
    auto out0 = _mm512_floor_ps(a0);
    auto C_bf16 = _mm512_cvtneps_pbh(out0);
    _mm256_mask_storeu_ps(dst + i, tail_mask, _mm256_castsi256_ps(C_bf16));
  }
}
#else
inline void floorbf_func(int len, int16_t* src, float* dst) { assert(0); }
#endif

inline void floorf_func(int len, float* src, float* dst) {
  int i = 0;
  int vec_size = 512 / 32;
  __mmask16 mask16 = 0xFFFF;
  for (; i <= len - vec_size; i += vec_size) {
    auto a1 = _mm512_loadu_ps(src + i);
    auto out1 = _mm512_floor_ps(a1);
    _mm512_mask_storeu_ps(dst + i, mask16, out1);
  }
  if (len - i) {
    auto tail_mask = calculat_offset(len - i, vec_size);
    auto a1 = _mm512_maskz_loadu_ps(tail_mask, src + i);
    auto out1 = _mm512_floor_ps(a1);
    _mm512_mask_storeu_ps(dst + i, tail_mask, out1);
  }
}

inline void rsqrtf_func(int len, float* src, float* dst) {
  int i = 0;
  int vec_size = 512 / 32;
  __mmask16 mask16 = 0xFFFF;

  for (; i <= len - vec_size; i += vec_size) {
    auto a1 = _mm512_loadu_ps(src + i);
    auto out1 = _mm512_rsqrt14_ps(a1);
    _mm512_mask_storeu_ps(dst + i, mask16, out1);
  }
  if (len - i) {
    auto tail_mask = calculat_offset(len - i, vec_size);
    auto a1 = _mm512_maskz_loadu_ps(tail_mask, src + i);
    auto out1 = _mm512_rsqrt14_ps(a1);
    _mm512_mask_storeu_ps(dst + i, tail_mask, out1);
  }
}

#if defined(__GNUC__) && (__GNUC__ > 9)
inline void rsqrtbf_func(int len, int16_t* src, float* dst) {
  int i = 0;
  int vec_size = 512 / 16;
  __mmask16 mask16 = 0xFFFF;
  auto alpha_vec = _mm512_set1_ps(0.0);
  for (; i <= len - vec_size; i += vec_size) {
    auto a0 = _mm512_cvtbf16f32_load(mask16, src + i);
    auto a1 = _mm512_cvtbf16f32_load(mask16, src + i + 16);

    auto out0 = _mm512_rsqrt14_ps(a0);
    auto out1 = _mm512_rsqrt14_ps(a1);

    auto C_bf16 = _mm512_cvtne2ps_pbh(out1, out0);
    _mm512_mask_storeu_ps(dst + i / 2, mask16, _mm512_castsi512_ps(C_bf16));
  }
  if ((len - i) > 16) {
    auto a0 = _mm512_cvtbf16f32_load(mask16, src + i);
    auto out0 = _mm512_rsqrt14_ps(a0);
    auto C_bf16 = _mm512_cvtneps_pbh(out0);
    _mm256_storeu_ps(dst + i / 2, _mm256_castsi256_ps(C_bf16));
    i += 16;
  }
  if (len - i) {
    auto tail_mask = calculat_offset(len - i, vec_size);
    auto a0 = _mm512_cvtbf16f32_load(tail_mask, src + i);
    auto out0 = _mm512_rsqrt14_ps(a0);
    auto C_bf16 = _mm512_cvtneps_pbh(out0);
    _mm256_mask_storeu_ps(dst + i, tail_mask, _mm256_castsi256_ps(C_bf16));
  }
}
#elif defined(__GNUC__) && (__GNUC__ > 8)
inline void rsqrtbf_func(int len, int16_t* src, float* dst) {
  int i = 0;
  int vec_size = 512 / 32;
  __mmask16 mask16 = 0xFFFF;
  auto alpha_vec = _mm512_set1_ps(0.0);
  auto tail_mask = calculat_offset(len, vec_size);
  for (; i <= len - vec_size; i += vec_size) {
    auto a0 = _mm512_cvtbf16f32_load(mask16, src + i);
    auto out0 = _mm512_rsqrt14_ps(a0);
    auto C_bf16 = _mm512_cvtneps_pbh(out0);
    _mm256_storeu_ps(dst + i, _mm256_castsi256_ps(C_bf16));
  }
  if (len - i) {
    auto a0 = _mm512_cvtbf16f32_load(tail_mask, src + i);
    auto out0 = _mm512_rsqrt14_ps(a0);
    auto C_bf16 = _mm512_cvtneps_pbh(out0);
    _mm256_mask_storeu_ps(dst + i, tail_mask, _mm256_castsi256_ps(C_bf16));
  }
}
#else
inline void rsqrtbf_func(int len, int16_t* src, float* dst) {}
#endif

#if defined(__GNUC__) && (__GNUC__ > 9)
static inline __m512 pexp(const __m512& _x) {
  __m512 p16f_1 = _mm512_set1_ps(1.0f);
  __m512 p16f_half = _mm512_set1_ps(0.5f);
  __m512 p16f_127 = _mm512_set1_ps(127.f);
  __m512 p16f_exp_hi = _mm512_set1_ps(88.3762626647950f);
  __m512 p16f_exp_lo = _mm512_set1_ps(-88.3762626647949f);

  __m512 p16f_cephes_LOG2EF = _mm512_set1_ps(1.44269504088896341f);

  __m512 p16f_cephes_exp_p0 = _mm512_set1_ps(1.9875691500E-4f);
  __m512 p16f_cephes_exp_p1 = _mm512_set1_ps(1.3981999507E-3f);
  __m512 p16f_cephes_exp_p2 = _mm512_set1_ps(8.3334519073E-3f);
  __m512 p16f_cephes_exp_p3 = _mm512_set1_ps(4.1665795894E-2f);
  __m512 p16f_cephes_exp_p4 = _mm512_set1_ps(1.6666665459E-1f);
  __m512 p16f_cephes_exp_p5 = _mm512_set1_ps(5.0000001201E-1f);

  // Clamp x.
  __m512 x = _mm512_max_ps(_mm512_min_ps(_x, p16f_exp_hi), p16f_exp_lo);

  // Express exp(x) as exp(m*ln(2) + r), start by extracting
  // m = floor(x/ln(2) + 0.5).
  __m512 m = _mm512_floor_ps(_mm512_fmadd_ps(x, p16f_cephes_LOG2EF, p16f_half));

  // Get r = x - m*ln(2). If no FMA instructions are available, m*ln(2) is
  // subtracted out in two parts, m*C1+m*C2 = m*ln(2), to avoid accumulating
  // truncation errors. Note that we don't use the "pmadd" function here to
  // ensure that a precision-preserving FMA instruction is used.
  __m512 p16f_nln2 = _mm512_set1_ps(-0.6931471805599453f);
  __m512 r = _mm512_fmadd_ps(m, p16f_nln2, x);

  __m512 r2 = _mm512_mul_ps(r, r);

  // TODO(gonnet): Split into odd/even polynomials and try to exploit
  //               instruction-level parallelism.
  __m512 y = p16f_cephes_exp_p0;
  y = _mm512_fmadd_ps(y, r, p16f_cephes_exp_p1);
  y = _mm512_fmadd_ps(y, r, p16f_cephes_exp_p2);
  y = _mm512_fmadd_ps(y, r, p16f_cephes_exp_p3);
  y = _mm512_fmadd_ps(y, r, p16f_cephes_exp_p4);
  y = _mm512_fmadd_ps(y, r, p16f_cephes_exp_p5);
  y = _mm512_fmadd_ps(y, r2, r);
  y = _mm512_add_ps(y, p16f_1);

  // Build emm0 = 2^m.
  __m512i emm0 = _mm512_cvttps_epi32(_mm512_add_ps(m, p16f_127));
  emm0 = _mm512_slli_epi32(emm0, 23);

  // Return 2^m * exp(r).
  return _mm512_max_ps(_mm512_mul_ps(y, _mm512_castsi512_ps(emm0)), _x);
};

static inline __m512 erf_avx512(const __m512& src512) {
  const __m512 coeff0 = _mm512_set1_ps(+7.853861353153693E-5);
  const __m512 coeff1 = _mm512_set1_ps(-8.010193625184903E-4);
  const __m512 coeff2 = _mm512_set1_ps(+5.188327685732524E-3);
  const __m512 coeff3 = _mm512_set1_ps(-2.685381193529856E-2);
  const __m512 coeff4 = _mm512_set1_ps(+1.128358514861418E-1);
  const __m512 coeff5 = _mm512_set1_ps(-3.761262582423300E-1);
  const __m512 coeff6 = _mm512_set1_ps(+1.128379165726710E+0);

  __m512 dst512;
  __m512 base512 = _mm512_mul_ps(src512, src512);
  dst512 = _mm512_fmadd_ps(coeff0, base512, coeff1);
  dst512 = _mm512_fmadd_ps(dst512, base512, coeff2);
  dst512 = _mm512_fmadd_ps(dst512, base512, coeff3);
  dst512 = _mm512_fmadd_ps(dst512, base512, coeff4);
  dst512 = _mm512_fmadd_ps(dst512, base512, coeff5);
  dst512 = _mm512_fmadd_ps(dst512, base512, coeff6);
  dst512 = _mm512_mul_ps(dst512, src512);

  return dst512;
}

static inline __m512 erfc_avx512(const __m512& src512) {
  const __m512 Pcoeff0 = _mm512_set1_ps(+2.326819970068386E-2);
  const __m512 Pcoeff1 = _mm512_set1_ps(-1.387039388740657E-1);
  const __m512 Pcoeff2 = _mm512_set1_ps(+3.687424674597105E-1);
  const __m512 Pcoeff3 = _mm512_set1_ps(-5.824733027278666E-1);
  const __m512 Pcoeff4 = _mm512_set1_ps(+6.210004621745983E-1);
  const __m512 Pcoeff5 = _mm512_set1_ps(-4.944515323274145E-1);
  const __m512 Pcoeff6 = _mm512_set1_ps(+3.404879937665872E-1);
  const __m512 Pcoeff7 = _mm512_set1_ps(-2.741127028184656E-1);
  const __m512 Pcoeff8 = _mm512_set1_ps(+5.638259427386472E-1);

  const __m512 Rcoeff0 = _mm512_set1_ps(-1.047766399936249E+1);
  const __m512 Rcoeff1 = _mm512_set1_ps(+1.297719955372516E+1);
  const __m512 Rcoeff2 = _mm512_set1_ps(-7.495518717768503E+0);
  const __m512 Rcoeff3 = _mm512_set1_ps(+2.921019019210786E+0);
  const __m512 Rcoeff4 = _mm512_set1_ps(-1.015265279202700E+0);
  const __m512 Rcoeff5 = _mm512_set1_ps(+4.218463358204948E-1);
  const __m512 Rcoeff6 = _mm512_set1_ps(-2.820767439740514E-1);
  const __m512 Rcoeff7 = _mm512_set1_ps(+5.641895067754075E-1);

  const __m512 one = _mm512_set1_ps(1.0);
  const __m512 two = _mm512_set1_ps(2.0);
  const __m512 zero = _mm512_set1_ps(0.0);
  const __m512 MinorMaxlog = _mm512_set1_ps(-88.72283905206835);

  __m512 abssrc = _mm512_abs_ps(src512);
  __m512 nabssrc = _mm512_sub_ps(zero, abssrc);
  __m512 v = _mm512_mul_ps(abssrc, nabssrc);
  __m512 z = pexp(v);
  __m512 q = _mm512_div_ps(one, abssrc);
  __m512 y = _mm512_mul_ps(q, q);

  __mmask16 PCoeff_mask = _mm512_cmp_ps_mask(abssrc, two, _CMP_LT_OQ); // < 2
  __mmask16 RCoeff_mask = ~PCoeff_mask;

  __m512 pP;
  __m512 pR;
  if (PCoeff_mask) {
    pP = _mm512_fmadd_ps(Pcoeff0, y, Pcoeff1);
    pP = _mm512_fmadd_ps(pP, y, Pcoeff2);
    pP = _mm512_fmadd_ps(pP, y, Pcoeff3);
    pP = _mm512_fmadd_ps(pP, y, Pcoeff4);
    pP = _mm512_fmadd_ps(pP, y, Pcoeff5);
    pP = _mm512_fmadd_ps(pP, y, Pcoeff6);
    pP = _mm512_fmadd_ps(pP, y, Pcoeff7);
    pP = _mm512_fmadd_ps(pP, y, Pcoeff8);
  }

  if (RCoeff_mask) {
    pR = _mm512_fmadd_ps(Rcoeff0, y, Rcoeff1);
    pR = _mm512_fmadd_ps(pR, y, Rcoeff2);
    pR = _mm512_fmadd_ps(pR, y, Rcoeff3);
    pR = _mm512_fmadd_ps(pR, y, Rcoeff4);
    pR = _mm512_fmadd_ps(pR, y, Rcoeff5);
    pR = _mm512_fmadd_ps(pR, y, Rcoeff6);
    pR = _mm512_fmadd_ps(pR, y, Rcoeff7);
  }

  pP = _mm512_mask_mov_ps(pP, RCoeff_mask, pR);
  //  y = z * q * p;
  //  float y_clamp = z < -kMaxlog ? 0 : y;

  //  return x < 0 ? 2 - y_clamp : y_clamp;
  y = _mm512_mul_ps(z, q);
  y = _mm512_mul_ps(y, pP);
  __mmask16 y_clamp_mask = _mm512_cmp_ps_mask(z, MinorMaxlog, _CMP_LT_OQ);
  __m512 y_clamp = _mm512_mask_mov_ps(y, y_clamp_mask, zero);
  __mmask16 x_mask = _mm512_cmp_ps_mask(src512, zero, _CMP_LT_OQ);
  __m512 y_clamp2 = _mm512_sub_ps(two, y_clamp);
  y = _mm512_mask_mov_ps(y_clamp, x_mask, y_clamp2);
  y = _mm512_sub_ps(one, y);

  return y;
}
#endif

template <typename T, typename Q>
inline void splitter(const T& n, const Q& team, const Q& tid, T& n_start,
                     T& n_end) {
  if (team <= 1 || n == 0) {
    n_start = 0;
    n_end = n;
  } else {
    T n1 = (n + (T)team - 1) / (T)team;
    T n2 = n1 - 1;
    T T1 = n - n2 * (T)team;
    n_end = (T)tid < T1 ? n1 : n2;
    n_start = (T)tid <= T1 ? tid * n1 : T1 * n1 + ((T)tid - T1) * n2;
  }
  n_end += n_start;
}
template <typename T0, typename F>
void for_1d(const int& ithr, const int& nthr, const T0& D0, const F& func) {
  T0 d0{0}, end{0};
  splitter(D0, nthr, ithr, d0, end);
  for (; d0 < end; ++d0) func(d0);
}
template <typename T0, typename F>
void parallel_for(const T0& D0, const F& func) {
#pragma omp parallel
  for_1d(omp_get_thread_num(), omp_get_num_threads(), D0, func);
}
inline bool parallel_it_step() { return true; }
template <typename Q, typename R, typename... Args>
inline bool parallel_it_step(Q& x, const R& X, Args&&... tuple) {
  if (parallel_it_step(static_cast<Args>(tuple)...)) {
    x = (x + 1) % X;
    return x == 0;
  }
  return false;
}

template <typename T>
inline T parallel_it_init(T start) {
  return start;
}
template <typename T, typename Q, typename R, typename... Args>
inline T parallel_it_init(T start, Q& x, const R& X, Args&&... tuple) {
  start = parallel_it_init(start, static_cast<Args>(tuple)...);
  x = start % X;
  return start / X;
}

template <typename T0, typename T1, typename F>
void for_2d(const int& ithr, const int& nthr, const T0& D0, const T1& D1,
            const F& func) {
  const size_t work_amount = (size_t)D0 * D1;
  if (work_amount == 0) return;
  size_t start{0}, end{0};
  splitter(work_amount, nthr, ithr, start, end);

  T0 d0{0};
  T1 d1{0};
  parallel_it_init(start, d0, D0, d1, D1);
  for (size_t iwork = start; iwork < end; ++iwork) {
    func(d0, d1);
    parallel_it_step(d0, D0, d1, D1);
  }
}
template <typename T0, typename T1, typename F>
void parallel_for2d(const T0& D0, const T1& D1, const F& func) {
#pragma omp parallel
  for_2d(omp_get_thread_num(), omp_get_num_threads(), D0, D1, func);
}
const int block_size = 16;
typedef __m512 vec_type_f;
typedef __m512i vec_type_i;
typedef __mmask16 vmask_type;
using SizeVector = std::vector<int>;
inline int count(SizeVector dims, int start_ind, int end_ind) {
  size_t count = 1;
  for (size_t i = start_ind; i < end_ind; i++) count *= dims[i];
  return static_cast<int>(count);
}

inline int count(SizeVector dims, size_t start_ind = 0) {
  return count(dims, start_ind, dims.size());
}
static inline void _mm_uni_storeu_ps(float* pdst, const __m512& vec) {
  _mm512_storeu_ps(pdst, vec);
}
static inline void _mm_uni_storeu_si(void* pdst, const __m512i vec) {
  _mm512_storeu_si512(pdst, vec);
}
static inline __mmask16 _mm_uni_cmpgt_i32(__m512i vec0, __m512i vec1) {
  return _mm512_cmp_epi32_mask(vec1, vec0, 1);
}
static inline __mmask16 _mm_uni_cmpgt_ps(__m512 vec0, __m512 vec1) {
  return _mm512_cmp_ps_mask(vec0, vec1, 14);
}
static inline __m512 _mm_uni_any_ps() { return __m512{}; }
static inline __m512i _mm_uni_any_epi32() { return __m512i{}; }
static inline __m512i _mm_uni_set1_epi32(int value) {
  return _mm512_mask_set1_epi32(_mm_uni_any_epi32(), (__mmask16)-1, value);
}
static inline __m512i _mm_uni_setzero_si() { return _mm512_setzero_si512(); }
static inline __m512 _mm_uni_blendv_ps(__m512 vec0, __m512 vec1, __m512 vmask) {
  return _mm512_mask_blend_ps(
      _mm512_cmpneq_epi32_mask(_mm512_castps_si512(vmask),
                               _mm_uni_set1_epi32(0)),
      vec0, vec1);
}

static inline __m512 _mm_uni_blendv_ps(__m512 vec0, __m512 vec1,
                                       __mmask16 vmask) {
  return _mm512_mask_blend_ps(vmask, vec0, vec1);
}
struct cmpgt_ps {
  static inline vmask_type cmp_ps(const __m512 _Left, const __m512 _Right) {
    return _mm_uni_cmpgt_ps(_Left, _Right);
  }
};

struct cmplt_ps {
  static inline vmask_type cmp_ps(const __m512 _Left, const __m512 _Right) {
    return _mm_uni_cmpgt_ps(_Right, _Left);
  }
};
static inline __m512 _mm_uni_loadu_ps(const float* psrc) {
  return _mm512_mask_loadu_ps(_mm_uni_any_ps(), (__mmask16)-1, psrc);
}

template <class Compare1, template <typename> class Compare2>
void top1_axis(const float* src_data, float* dst_data, int* dst_idx,
               SizeVector in_dims, int32_t axis, int before_num, int dim,
               int src_k, int count_vec, bool sort_value) {
  int after_num = count(in_dims, axis + 1, in_dims.size());
  int first_index = 0;
  parallel_for2d(before_num, after_num / block_size, [&](int i0, int ib1) {
    int s_index = i0 * dim * after_num + ib1 * block_size;
    vec_type_f vmax_val = _mm_uni_loadu_ps(src_data + s_index);
    vec_type_i vindex_max_val = _mm_uni_setzero_si();
    for (int i2 = 1; i2 < dim; i2++) {
      s_index += after_num;
      vec_type_f vsrc = _mm_uni_loadu_ps(src_data + s_index);
      vmask_type vmask = Compare1::cmp_ps(vsrc, vmax_val);
      vmax_val = _mm_uni_blendv_ps(vmax_val, vsrc, vmask);

      vec_type_i vindex_cur_val = _mm_uni_set1_epi32(i2);

      vindex_max_val =
          _mm512_mask_blend_epi32(vmask, vindex_max_val, vindex_cur_val);
    }
    if (dst_data)
      _mm_uni_storeu_ps(dst_data + i0 * after_num + ib1 * block_size, vmax_val);
    if (dst_idx)
      _mm_uni_storeu_si(reinterpret_cast<vec_type_i*>(dst_idx + i0 * after_num +
                                                      ib1 * block_size),
                        vindex_max_val);
  });
  first_index = after_num / block_size * block_size;
  int rest = after_num - first_index;
  parallel_for2d(before_num, rest, [&](int i0, int i1) {
    int index_max_val = 0;
    int s_index = i0 * dim * after_num + first_index + i1;
    float max_val = src_data[s_index];
    for (int i2 = 1; i2 < dim; i2++) {
      s_index += after_num;
      if (Compare2<float>()(src_data[s_index], max_val)) {
        max_val = src_data[s_index];
        index_max_val = i2;
      }
    }
    if (dst_data) dst_data[i0 * after_num + first_index + i1] = max_val;
    if (dst_idx) dst_idx[i0 * after_num + first_index + i1] = index_max_val;
  });
}

template <template <typename> class Compare>
void top1(const float* src_data, float* dst_data, int* dst_idx,
          SizeVector in_dims, int32_t axis, int before_num, int dim, int src_k,
          int count_vec, bool sort_value) {
  parallel_for(before_num, [&](int i0) {
    int index_max_val = 0;
    int s_index = i0 * dim;
    float max_val = src_data[s_index];
    for (int i1 = 1; i1 < dim; i1++) {
      s_index++;
      if (Compare<float>()(src_data[s_index], max_val)) {
        max_val = src_data[s_index];
        index_max_val = i1;
      }
    }
    if (dst_data) dst_data[i0] = max_val;
    if (dst_idx) dst_idx[i0] = index_max_val;
  });
}

template <class Compare1, template <typename> class Compare2>
void topk_axis(const float* src_data, float* dst_data, int* dst_idx,
               SizeVector in_dims, int32_t axis, int before_num, int dim,
               int src_k, int count_vec, bool sort_value) {
  int after_num = count(in_dims, axis + 1, in_dims.size());
  int first_index = 0;

  if (src_k < count_vec) {
    parallel_for2d(before_num, after_num / block_size, [&](int i0, int ib1) {
      const int N = 32;
      vec_type_f vmax_values[N];
      vec_type_i vmax_indexes[N];
      vec_type_f vtmp;
      vec_type_i vtmp_indexes;
      vmask_type vmask;
      int s_index = i0 * dim * after_num + ib1 * block_size;

      auto vswap_func = [&](int index1, int index2) {
        vtmp = vmax_values[index1];
        vmax_values[index1] =
            _mm_uni_blendv_ps(vmax_values[index1], vmax_values[index2], vmask);
        vmax_values[index2] =
            _mm_uni_blendv_ps(vmax_values[index2], vtmp, vmask);

        vtmp_indexes = vmax_indexes[index1];

        vmax_indexes[index1] = _mm512_mask_blend_epi32(
            vmask, vmax_indexes[index1], vmax_indexes[index2]);
        vmax_indexes[index2] =
            _mm512_mask_blend_epi32(vmask, vmax_indexes[index2], vtmp_indexes);
      };

      for (int i2 = 0; i2 < src_k; i2++) {
        vmax_values[i2] = _mm_uni_loadu_ps(src_data + s_index);
        vmax_indexes[i2] = _mm_uni_set1_epi32(i2);
        s_index += after_num;
      }
      for (int i2 = 0; i2 < src_k - 1; i2++) {
        for (int i3 = src_k - 1; i3 > i2; i3--) {
          vmask = Compare1::cmp_ps(vmax_values[i3], vmax_values[i3 - 1]);
          if (vmask) vswap_func(i3, i3 - 1);
        }
      }
      for (int i2 = src_k; i2 < dim; i2++) {
        vmax_values[src_k] = _mm_uni_loadu_ps(src_data + s_index);
        vmax_indexes[src_k] = _mm_uni_set1_epi32(i2);
        for (int i3 = src_k; i3 > 0; i3--) {
          vmask = Compare1::cmp_ps(vmax_values[i3], vmax_values[i3 - 1]);
          if (vmask)
            vswap_func(i3, i3 - 1);
          else
            break;
        }
        s_index += after_num;
      }
      if (!sort_value) {
        for (int i2 = 0; i2 < src_k - 1; i2++) {
          for (int i3 = src_k - 1; i3 > i2; i3--) {
            vmask = _mm_uni_cmpgt_i32(vmax_indexes[i3 - 1], vmax_indexes[i3]);
            if (vmask)
              vswap_func(i3, i3 - 1);
            else
              break;
          }
        }
      }
      if (dst_data) {
        for (int i2 = 0; i2 < src_k; i2++)
          _mm_uni_storeu_ps(
              dst_data + (i0 * src_k + i2) * after_num + ib1 * block_size,
              vmax_values[i2]);
      }
      if (dst_idx) {
        for (int i2 = 0; i2 < src_k; i2++)
          _mm_uni_storeu_si(
              reinterpret_cast<vec_type_i*>(
                  dst_idx + (i0 * src_k + i2) * after_num + ib1 * block_size),
              vmax_indexes[i2]);
      }
    });
    first_index = after_num / block_size * block_size;
  }
  int rest = after_num - first_index;
  parallel_for2d(before_num, rest, [&](int i0, int i1) {
    std::vector<float> max_values(src_k + 1);
    std::vector<int> max_indexes(src_k + 1);
    float tmp_value;
    int tmp_index;
    int s_index = i0 * dim * after_num + first_index + i1;

    auto swap_func = [&](int index1, int index2) {
      tmp_value = max_values[index1];
      max_values[index1] = max_values[index2];
      max_values[index2] = tmp_value;

      tmp_index = max_indexes[index1];
      max_indexes[index1] = max_indexes[index2];
      max_indexes[index2] = tmp_index;
    };

    for (int i2 = 0; i2 < src_k; i2++) {
      max_values[i2] = src_data[s_index];
      max_indexes[i2] = i2;
      s_index += after_num;
    }
    for (int i2 = 0; i2 < src_k - 1; i2++) {
      for (int i3 = src_k - 1; i3 > i2; i3--) {
        if (Compare2<float>()(max_values[i3], max_values[i3 - 1])) {
          swap_func(i3, i3 - 1);
        }
      }
    }
    for (int i2 = src_k; i2 < dim; i2++) {
      max_values[src_k] = src_data[s_index];
      max_indexes[src_k] = i2;
      for (int i3 = src_k; i3 > 0; i3--) {
        if (Compare2<float>()(max_values[i3], max_values[i3 - 1]))
          swap_func(i3, i3 - 1);
        else
          break;
      }
      s_index += after_num;
    }
    if (!sort_value) {
      for (int i2 = 0; i2 < src_k - 1; i2++) {
        for (int i3 = src_k - 1; i3 > i2; i3--) {
          if (std::greater<int>()(max_indexes[i3 - 1], max_indexes[i3])) {
            swap_func(i3, i3 - 1);
          }
        }
      }
    }
    if (dst_data) {
      for (int i2 = 0; i2 < src_k; i2++)
        dst_data[i0 * src_k * after_num + i2 * after_num + first_index + i1] =
            max_values[i2];
    }
    if (dst_idx) {
      for (int i2 = 0; i2 < src_k; i2++)
        dst_idx[i0 * src_k * after_num + i2 * after_num + first_index + i1] =
            max_indexes[i2];
    }
  });
}

template <template <typename> class Compare>
void topk(const float* src_data, float* dst_data, int* dst_idx,
          SizeVector in_dims, int32_t axis, int before_num, int dim, int src_k,
          int count_vec, bool sort_value) {
  parallel_for(before_num, [&](int i0) {
    std::vector<float> max_values(src_k + 1);
    std::vector<int> max_indexes(src_k + 1);
    float tmp_value;
    int tmp_index;
    int s_index = i0 * dim;

    auto swap_func = [&](int index1, int index2) {
      tmp_value = max_values[index1];
      max_values[index1] = max_values[index2];
      max_values[index2] = tmp_value;

      tmp_index = max_indexes[index1];
      max_indexes[index1] = max_indexes[index2];
      max_indexes[index2] = tmp_index;
    };

    for (int i2 = 0; i2 < src_k; i2++) {
      max_values[i2] = src_data[s_index];
      max_indexes[i2] = i2;
      s_index++;
    }
    for (int i2 = 0; i2 < src_k - 1; i2++) {
      for (int i3 = src_k - 1; i3 > i2; i3--) {
        if (Compare<float>()(max_values[i3], max_values[i3 - 1])) {
          swap_func(i3, i3 - 1);
        }
      }
    }
    for (int i2 = src_k; i2 < dim; i2++) {
      max_values[src_k] = src_data[s_index];
      max_indexes[src_k] = i2;
      for (int i3 = src_k; i3 > 0; i3--) {
        if (Compare<float>()(max_values[i3], max_values[i3 - 1]))
          swap_func(i3, i3 - 1);
        else
          break;
      }
      s_index++;
    }
    if (!sort_value) {
      for (int i2 = 0; i2 < src_k - 1; i2++) {
        for (int i3 = src_k - 1; i3 > i2; i3--) {
          if (std::greater<int>()(max_indexes[i3 - 1], max_indexes[i3])) {
            swap_func(i3, i3 - 1);
          }
        }
      }
    }
    if (dst_data) {
      for (int i2 = 0; i2 < src_k; i2++)
        dst_data[i0 * src_k + i2] = max_values[i2];
    }
    if (dst_idx) {
      for (int i2 = 0; i2 < src_k; i2++)
        dst_idx[i0 * src_k + i2] = max_indexes[i2];
    }
  });
}

void topk_func(float* src, float* dst_data, int* dst_idx,
               std::vector<int32_t> src_dims, uint32_t K, bool largest,
               bool sorted, uint32_t axis) {
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
  if (axis_ < 0) axis_ += src_dims.size();
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
    if (src_dims[j] != 1) break;
  }
  if (static_cast<size_t>(j) == axis) is_last_dim = true;

  for (size_t i = 0; i < axis; i++) {
    axis_step *= src_dims[i];
  }
  axis_dim = src_dims[axis];
  for (size_t i = (axis + 1); i < src_dims.size(); i++) {
    axis_stride *= src_dims[i];
  }
  dim = static_cast<int>(src_dims[axis]);
  before_num = count(src_dims, 0, axis);
  if (src_k == 1) {
    if (is_last_dim) {
      if (mode_max)
        top1<std::greater>(src, dst_data, dst_idx, in_dims, axis, before_num,
                           dim, src_k, count_vec, sort_value);
      else
        top1<std::less>(src, dst_data, dst_idx, in_dims, axis, before_num, dim,
                        src_k, count_vec, sort_value);
    } else {
      if (mode_max)
        top1_axis<cmpgt_ps, std::greater>(src, dst_data, dst_idx, in_dims, axis,
                                          before_num, dim, src_k, count_vec,
                                          sort_value);
      else
        top1_axis<cmplt_ps, std::less>(src, dst_data, dst_idx, in_dims, axis,
                                       before_num, dim, src_k, count_vec,
                                       sort_value);
    }
  } else {
    if (is_last_dim) {
      if (mode_max) {
        topk<std::greater>(src, dst_data, dst_idx, in_dims, axis, before_num,
                           dim, src_k, count_vec, sort_value);
      } else
        topk<std::less>(src, dst_data, dst_idx, in_dims, axis, before_num, dim,
                        src_k, count_vec, sort_value);
    } else {
      if (mode_max)
        topk_axis<cmpgt_ps, std::greater>(src, dst_data, dst_idx, in_dims, axis,
                                          before_num, dim, src_k, count_vec,
                                          sort_value);
      else
        topk_axis<cmplt_ps, std::less>(src, dst_data, dst_idx, in_dims, axis,
                                       before_num, dim, src_k, count_vec,
                                       sort_value);
    }
  }
}

#if defined(__GNUC__) && (__GNUC__ > 9)
static __m512 __mm512_fake_erf(__m512 src) {
  auto abssrc = _mm512_abs_ps(src);
  __mmask16 erf_mask =
      _mm512_cmp_ps_mask(abssrc, _mm512_set1_ps(1.0), _CMP_LT_OQ); // < 1
  __m512 dst512 = erf_avx512(src);
  __m512 dstc512 = erfc_avx512(src);
  return _mm512_mask_blend_ps(erf_mask, dstc512, dst512);
}
static void erf_func(float* src, float* dst, size_t len) {
  int i;
  for (i = 0; i + 16 <= len; i += 16) {
    __m512 src512 = _mm512_loadu_ps(src + i);
    __m512 abssrc = _mm512_abs_ps(src512);
    __mmask16 erf_mask =
        _mm512_cmp_ps_mask(abssrc, _mm512_set1_ps(1.0), _CMP_LT_OQ); // < 1
    __mmask16 erfc_mask = ~erf_mask;
    auto dst512 = __mm512_fake_erf(src512);
    _mm512_storeu_ps(dst + i, dst512);
  }

  int remain = len - i;
  if (remain) {
    __mmask16 mask = 0xffff;
    mask = mask >> (16 - remain);
    __m512 src512 = _mm512_maskz_loadu_ps(mask, src + i);
    __mmask16 erf_mask =
        _mm512_cmp_ps_mask(src512, _mm512_set1_ps(1.0), _CMP_LT_OQ); // < 1
    __mmask16 erfc_mask = ~erf_mask;
    auto dst512 = __mm512_fake_erf(src512);
    _mm512_mask_storeu_ps(dst + i, mask, dst512);
    // printf("erf_p remain...\n");
  }
  return;
}
#else
static void erf_func(float* src, float* dst, size_t len) { assert(0); }
#endif

#if defined(__GNUC__) && (__GNUC__ > 9)
static void erf_bf16_func(int16_t* src, float* dst, size_t len) {
  int i = 0;
  int vec_size = 512 / 16;
  __mmask16 mask16 = 0xFFFF;
  for (; i <= len - vec_size; i += vec_size) {
    auto a0 = _mm512_cvtbf16f32_load(mask16, src + i);
    auto a1 = _mm512_cvtbf16f32_load(mask16, src + i + 16);
    auto erf_dst_a0 = __mm512_fake_erf(a0);
    auto erf_dst_a1 = __mm512_fake_erf(a1);
    auto C_bf16 = _mm512_cvtne2ps_pbh(erf_dst_a1, erf_dst_a0);
    _mm512_mask_storeu_ps(dst + i / 2, mask16, _mm512_castsi512_ps(C_bf16));
  }
  if ((len - i) > 16) {
    auto a0 = _mm512_cvtbf16f32_load(mask16, src + i);
    auto out0 = __mm512_fake_erf(a0);
    auto C_bf16 = _mm512_cvtneps_pbh(out0);
    _mm256_storeu_ps(dst + i / 2, _mm256_castsi256_ps(C_bf16));
    i += 16;
  }
  if (len - i) {
    auto tail_mask = calculat_offset(len - i, vec_size);
    auto a0 = _mm512_cvtbf16f32_load(tail_mask, src + i);
    auto out0 = __mm512_fake_erf(a0);
    auto C_bf16 = _mm512_cvtneps_pbh(out0);
    _mm256_mask_storeu_ps(dst + i, tail_mask, _mm256_castsi256_ps(C_bf16));
  }
  return;
}
#else
static void erf_bf16_func(int16_t* src, float* dst, size_t len) { assert(0); }
#endif

// nms function related
enum class boxEncoding { CORNER, CENTER };

struct filteredBoxes {
  float score;
  int class_index;
  int box_index;
  filteredBoxes() : score(0), class_index(0), box_index(0) {}
  filteredBoxes(float _score, int _class_index, int _box_index)
      : score(_score), class_index(_class_index), box_index(_box_index) {}
};

struct Box {
  float score;
  int class_index;
  int box_index;
  Box() {}
  Box(float _score, int _class_index, int _box_index)
      : score(_score), class_index(_class_index), box_index(_box_index) {}
};

void nms_func(float* boxes, float* scores, size_t batch_idx, size_t class_num,
              size_t num_boxes, size_t max_num_outputs, float score_threshold,
              float iou_threshold, int32_t* output_indices) {
  auto intersectionOverUnion = [](const float* boxesI, const float* boxesJ,
                                  boxEncoding boxEncodingType) {
    float yminI, xminI, ymaxI, xmaxI, yminJ, xminJ, ymaxJ, xmaxJ;
    if (boxEncodingType == boxEncoding::CENTER) {
      //  box format: x_center, y_center, width, height
      yminI = boxesI[1] - boxesI[3] / 2.f;
      xminI = boxesI[0] - boxesI[2] / 2.f;
      ymaxI = boxesI[1] + boxesI[3] / 2.f;
      xmaxI = boxesI[0] + boxesI[2] / 2.f;
      yminJ = boxesJ[1] - boxesJ[3] / 2.f;
      xminJ = boxesJ[0] - boxesJ[2] / 2.f;
      ymaxJ = boxesJ[1] + boxesJ[3] / 2.f;
      xmaxJ = boxesJ[0] + boxesJ[2] / 2.f;
    } else {
      //  box format: y1, x1, y2, x2
      yminI = (std::min)(boxesI[0], boxesI[2]);
      xminI = (std::min)(boxesI[1], boxesI[3]);
      ymaxI = (std::max)(boxesI[0], boxesI[2]);
      xmaxI = (std::max)(boxesI[1], boxesI[3]);
      yminJ = (std::min)(boxesJ[0], boxesJ[2]);
      xminJ = (std::min)(boxesJ[1], boxesJ[3]);
      ymaxJ = (std::max)(boxesJ[0], boxesJ[2]);
      xmaxJ = (std::max)(boxesJ[1], boxesJ[3]);
    }

    float areaI = (ymaxI - yminI) * (xmaxI - xminI);
    float areaJ = (ymaxJ - yminJ) * (xmaxJ - xminJ);
    if (areaI <= 0.f || areaJ <= 0.f) return 0.f;

    float intersection_area =
        (std::max)((std::min)(ymaxI, ymaxJ) - (std::max)(yminI, yminJ), 0.f) *
        (std::max)((std::min)(xmaxI, xmaxJ) - (std::max)(xminI, xminJ), 0.f);
    return intersection_area / (areaI + areaJ - intersection_area);
  };
  size_t numFiltBox;
  bool sort_result_descending = true;
  boxEncoding boxEncodingType = boxEncoding::CORNER;

  if (max_num_outputs == 0) {
    return;
  }

  std::vector<filteredBoxes> filtBoxes(num_boxes);

  std::vector<Box> sorted_boxes;
  for (int box_idx = 0; box_idx < num_boxes; box_idx++) {
    float* scores_ptr = scores + box_idx * class_num;
    int idx = std::max_element(scores_ptr, scores_ptr + class_num) - scores_ptr;
    float score = scores_ptr[idx];
    if (score > score_threshold) {
      sorted_boxes.emplace_back(Box(score, idx, box_idx));
    }
  }

  int io_selection_size = 0;
  if (sorted_boxes.size() > 0) {
    auto _compare = [](const Box l, const Box r) {
      return (l.score > r.score ||
              ((l.score == r.score) && (l.box_index < r.box_index)));
    };
    std::sort(sorted_boxes.begin(), sorted_boxes.end(), _compare);
    for (int i = 0; i < sorted_boxes.size(); i++) {
      auto score = sorted_boxes[i].score;
      auto idx = sorted_boxes[i].class_index;
      auto box_idx = sorted_boxes[i].box_index;
    }
    filtBoxes[0] =
        filteredBoxes(sorted_boxes[0].score, sorted_boxes[0].class_index,
                      sorted_boxes[0].box_index);
    io_selection_size++;
    for (size_t box_idx = 1; (box_idx < sorted_boxes.size()) &&
                             (io_selection_size < max_num_outputs);
         box_idx++) {
      bool box_is_selected = true;
      for (int idx = io_selection_size - 1; idx >= 0; idx--) {
        float iou = intersectionOverUnion(
            &boxes[sorted_boxes[box_idx].box_index * 4],
            &boxes[filtBoxes[idx].box_index * 4], boxEncodingType);
        if (iou >= iou_threshold) {
          box_is_selected = false;
          break;
        }
      }

      if (box_is_selected) {
        filtBoxes[io_selection_size] = filteredBoxes(
            sorted_boxes[box_idx].score, sorted_boxes[box_idx].class_index,
            sorted_boxes[box_idx].box_index);
        io_selection_size++;
      }
    }
  }
  numFiltBox = io_selection_size;
  memset(output_indices, max_num_outputs * 3, 0);
  memset(output_indices, max_num_outputs, batch_idx);
  for (size_t idx = 0; idx < numFiltBox; idx++) {
    output_indices[max_num_outputs + 2 * idx] = filtBoxes[idx].class_index;
    output_indices[max_num_outputs + 2 * idx + 1] = filtBoxes[idx].box_index;
  }
}

} // namespace dnnl_utils
