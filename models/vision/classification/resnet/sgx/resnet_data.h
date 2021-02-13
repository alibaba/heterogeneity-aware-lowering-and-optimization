//===- resnet_data.h ------------------------------------------------------===//
//
// Copyright (C) 2020-2021 Alibaba Group Holding Limited.
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

#ifndef TESTS_BENCHMARKS_ONNX_RESNET_DATA_H
#define TESTS_BENCHMARKS_ONNX_RESNET_DATA_H

static const float test_input[1 * 3 * 224 * 224] = {
#if TEST_SET == 1
#include "input_data_1.txt"
#elif TEST_SET == 2
#include "input_data_2.txt"
#else
#include "input_data_0.txt"
#endif
};

#ifdef TEST_BN_0
static const float test_output_ref[1 * 3 * 224 * 224] = {
#include "output_bn0.txt"
};
#else
static const float test_output_ref[1000] = {
#if TEST_SET == 1
#include "output_data_1.txt"
#elif TEST_SET == 2
#include "output_data_2.txt"
#else
#include "output_data_0.txt"
#endif
};
#endif

#endif // TESTS_BENCHMARKS_ONNX_RESNET_DATA_H