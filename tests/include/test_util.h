//===- test_util.h --------------------------------------------------------===//
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

#ifndef TESTS_BENCHMARKS_TEST_UTIL_H_
#define TESTS_BENCHMARKS_TEST_UTIL_H_

#ifdef __cplusplus
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#else
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#endif

#ifdef __cplusplus
template <typename T>
static bool is_nan(const T& x) {
  return x != x;
}

template <typename T>
static const T* to_nhwc(const T* src, T* dst, int N, int H, int W, int C = 3) {
  for (int n = 0; n < N; ++n)
    for (int c = 0; c < C; ++c)
      for (int s = 0, HW = H * W; s < HW; ++s)
        dst[n * HW * C + s * C + c] = src[n * C * HW + c * HW + s];
  return dst;
}

template <typename T>
bool Verify(const T* out, const T* out_ref, size_t n, float thre = 1e-5) {
  for (size_t i = 0; i < n; ++i) {
    bool nan_mismatch = (is_nan(out[i]) ^ is_nan(out_ref[i]));
    if (nan_mismatch || fabs(out[i] - out_ref[i]) > thre) {
      std::cout << "miscompare: [" << i << "]: " << out[i]
                << " expects: " << out_ref[i] << "\n";
      return false;
    }
  }
  return true;
}

template <typename T>
float EvalCosSim(const T* out, const T* out_ref, size_t n) {
  float norm_out = 0.f;
  float norm_out_ref = 0.f;
  float dot = 0.f;
  for (size_t i = 0; i < n; ++i) {
    dot += out[i] * out_ref[i];
    norm_out += out[i] * out[i];
    norm_out_ref += out_ref[i] * out_ref[i];
  }
  norm_out = std::sqrt(norm_out);
  norm_out_ref = std::sqrt(norm_out_ref);

  return dot / (norm_out * norm_out_ref);
}
typedef std::chrono::high_resolution_clock::time_point timestamp_t;
static timestamp_t Now() { return std::chrono::high_resolution_clock::now(); }

static double GetDuration(const timestamp_t& begin, const timestamp_t& end) {
  std::chrono::duration<double> seconds = end - begin;
  return seconds.count();
}
#else

static int is_nan(float x) { return x != x ? 1 : 0; }

int Verify(const float* out, const float* out_ref, size_t n, float thre) {
  for (size_t i = 0; i < n; ++i) {
    int nan_mismatch = (is_nan(out[i]) ^ is_nan(out_ref[i]));
    if (nan_mismatch || fabs(out[i] - out_ref[i]) > thre) {
      printf("miscompare: [%lu]: %f expects: %f\n", i, out[i], out_ref[i]);
      return 0;
    }
  }
  return 1;
}
typedef clock_t timestamp_t;
static timestamp_t Now() { return clock(); }

static double GetDuration(timestamp_t start, timestamp_t end) {
  return (double)(end - start) / (double)(CLOCKS_PER_SEC);
}

#endif

#endif // TESTS_BENCHMARKS_TEST_UTIL_H_