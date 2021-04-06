//===- face_detect_test.cc
//-------------------------------------------------===//
//
// Copyright (C) 2019-2021 Alibaba Group Holding Limited.
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

// clang-format off
// prepare model&dataset
// RUN: cd %S && python3 get_model_and_data.py --cfg config.yaml TEST.MODEL_FILE pose_hrnet_w32_256x256.pth && cd -

// Testing using ODLA Tensorrt
// RUN: %halo_compiler -target cxx -disable-broadcasting %S/pose_hrnet_w32_256x256.onnx -o %T/model.cc -entry-func-name=model -fuse-conv-bias -disable-code-format

// RUN: %cxx %flags -c %T/model.cc -I%odla_path/include -o %T/model.o
// RUN: %cxx %flags -c %include -I%T %s -o %t_main.o
// RUN: %cxx %t_main.o %T/model.o %T/model.bin %odla_link -lodla_tensorrt -o %t_dnnl.exe
// RUN: %t_dnnl.exe | FileCheck %s

// CHECK: Result verified

// clang-format on

#include <stdio.h>
#include <vector>
#include <fstream>

#include "test_util.h"

#include "model.h"
#include "input.in"
#include "output.in"

#ifndef COMPARE_ERROR
#define COMPARE_ERROR 1e-4
#endif

#define OUTPUT_SIZE (1 * 16 * 64 * 64)
#define INPUT_SIZE (1 * 3 * 256 * 256)

int main(int argc, char** argv) {
  float output[OUTPUT_SIZE];

  std::vector<float> data_nhwc(INPUT_SIZE);
  std::vector<float> output_v(std::begin(output_gold), std::end(output_gold));

  const float* input = data;
  if (argc > 1) {
    to_nhwc(data, data_nhwc.data(), 1, 256, 256, 3);
    input = data_nhwc.data();
  }

  model_init();
  model(input, output);
  model_fini();
#ifdef DEBUG_OUT
  std::ofstream of_out;

  of_out.open("out_debug.txt", std::ofstream::out);

  for (int i = 0; i < OUTPUT_SIZE; i++) {
    of_out << output[i] << std::endl;
  }
#endif
  auto check = [](const float* out, const std::vector<float>& gold) {
    float cos_sim = EvalCosSim(out, gold.data(), gold.size());
    if (Verify(out, gold.data(), gold.size(), COMPARE_ERROR) &&
        (cos_sim - 0.99) >= 0) {
      std::cout << "Result verified" << std::endl;
      return true;
    }
    std::cout << "Result Fail" << std::endl;
    exit(1);
    return false;
  };

  check(output, output_v);

  return 0;
}
