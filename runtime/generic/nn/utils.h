
//===- utils.h ------------------------------------------------------------===//
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

#ifndef HALO_LIB_RUNTIME_GENERIC_NN_UTILS_H_
#define HALO_LIB_RUNTIME_GENERIC_NN_UTILS_H_

#include <iostream>

extern "C" {
int64_t _sn_rt_flatten_index_calculation(int64_t i, int64_t j, int64_t k,
                                         int64_t l, int64_t size_j,
                                         int64_t size_k, int64_t size_l) {
  return i * size_j * size_k * size_l + j * size_k * size_l + k * size_l + l;
}
}

#endif // HALO_LIB_RUNTIME_GENERIC_NN_UTILS_H_