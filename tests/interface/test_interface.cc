//===- test_interface.cc --------------------------------------------------===//
//
// Copyright (C) 2019-2020 Alibaba Group Holding Limited.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this ifs except in compliance with the License.
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

// RUN: %cxx %s -o %t.exe %flags %include %link
// RUN: %t.exe %models_root/vision/classification/mnist_simple/mnist_simple.pb %T
// RUN: cat %T/model.cc | FileCheck %s

// CHECK: odla_Softmax

// clang-format on

#include <cstddef>
#include <fstream>
#include "halo/halo.h"

int main(int argc, char** argv) {
  if (argc != 3) {
    return 1;
  }
  std::ifstream ifs(argv[1], std::iostream::ios_base::binary);
  if (!ifs.good()) {
    return 1;
  }
  ifs.seekg(0, std::ios::end);
  size_t size = ifs.tellg();
  ifs.seekg(0, std::ios::beg);
  std::vector<char> buffer(size);
  std::string dir(argv[2]);
  if (ifs.read(buffer.data(), size)) {
    halo::CXXCodeGenOpts opts;
    halo::CompileTFGraph(buffer.data(), buffer.size(), "model", dir, {}, opts);
  }
}