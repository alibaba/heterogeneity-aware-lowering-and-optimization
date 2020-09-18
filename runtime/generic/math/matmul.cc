//===- matmul.cc ----------------------------------------------------------===//
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
/// A dummy implementation.
void _sn_rt_matmul_f32(float* C, const float* A, const float* B, int64_t A_row,
                       int64_t A_col, int64_t B_row, int64_t B_col,
                       bool transposeA, bool transposeB) {
  auto C_row = (transposeA ? A_col : A_row);
  auto C_col = (transposeB ? B_row : B_col);
  if (!transposeA && transposeB) {
    for (int64_t i = 0; i < C_row; ++i) {
      for (int64_t j = 0; j < C_col; ++j) {
        *C = 0;
        for (int64_t k = 0; k < B_col; ++k)
          *C += A[i * A_col + k] * B[j * B_col + k];
        ++C;
      }
    }
    return;
  }
  if (transposeA && !transposeB) {
    for (int64_t i = 0; i < C_row; ++i) {
      for (int64_t j = 0; j < C_col; ++j) {
        *C = 0;
        for (int64_t k = 0; k < B_row; ++k)
          *C += A[k * A_col + i] * B[k * B_col + j];
        ++C;
      }
    }
    return;
  }
  if (!transposeA && !transposeB) {
    for (int64_t i = 0; i < C_row; ++i) {
      for (int64_t j = 0; j < C_col; ++j) {
        *C = 0;
        for (int64_t k = 0; k < B_row; ++k)
          *C += A[i * A_col + k] * B[k * B_col + j];
        ++C;
      }
    }
  }
}

void _sn_rt_gemm_f32(float* result, const float* A, const float* B,
                     const float* C, int64_t A_row, int64_t A_col,
                     int64_t B_row, int64_t B_col, int64_t C_noe,
                     bool transposeA, bool transposeB, float alpha,
                     float beta) {
  _sn_rt_matmul_f32(result, A, B, A_row, A_col, B_row, B_col, transposeA,
                    transposeB);
  auto result_row = (transposeA ? A_col : A_row);
  auto result_col = (transposeB ? B_row : B_col);
  if (C_noe == result_row * result_col) {
    for (int64_t i = 0; i < result_row; ++i) {
      for (int64_t j = 0; j < result_col; ++j) {
        *result *= alpha;
        *result += beta * C[i * result_col + j];
        ++result;
      }
    }
    return;
  }
  if (C_noe == 1) {
    for (int64_t i = 0; i < result_row; ++i) {
      for (int64_t j = 0; j < result_col; ++j) {
        *result *= alpha;
        *result += beta * (*C);
        ++result;
      }
    }
    return;
  }
  if (C_noe == result_row) {
    for (int64_t i = 0; i < result_row; ++i) {
      for (int64_t j = 0; j < result_col; ++j) {
        *result *= alpha;
        *result += beta * C[i];
        ++result;
      }
    }
    return;
  }
  if (C_noe == result_col) {
    for (int64_t i = 0; i < result_row; ++i) {
      for (int64_t j = 0; j < result_col; ++j) {
        *result *= alpha;
        *result += beta * C[j];
        ++result;
      }
    }
    return;
  }
}

void _sn_rt_batch_matmul_f32(float* C, const float* A, const float* B,
                             int64_t batches, int64_t A_row, int64_t A_col,
                             int64_t B_row, int64_t B_col, bool transposeA,
                             bool transposeB) {
  auto C_row = (transposeA ? A_col : A_row);
  auto C_col = (transposeB ? B_row : B_col);
  auto C_stride = C_row * C_col;
  auto A_stride = A_row * A_col;
  auto B_stride = B_row * B_col;
  for (int64_t i = 0, offset_c = 0, offset_a = 0, offset_b = 0; i < batches;
       ++i, offset_c += C_stride, offset_a += A_stride, offset_b += B_stride) {
    _sn_rt_matmul_f32(&C[offset_c], &A[offset_a], &B[offset_b], A_row, A_col,
                      B_row, B_col, transposeA, transposeB);
  }
}
}