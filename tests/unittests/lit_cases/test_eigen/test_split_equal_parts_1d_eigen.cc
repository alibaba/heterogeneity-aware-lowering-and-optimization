//===-test_split_equal_parts_1d_eigen.cc-----------------------------------------------------------===//
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

// clang-format off
// Testing CXX Code Gen using ODLA API on eigen
// RUN: %halo_compiler -target cxx -batch-size 1 %halo_compile_flags %data_path/test_split_equal_parts_1d/model.onnx -o %t.cc
// RUN: %cxx -c -fPIC -o %t.o %t.cc -I%odla_path/include
// RUN: %cxx -g %s %onnx_path/onnx.pb.cc %t.o %t.bin -I%T -I%odla_path/include -I%unittests_path -I%onnx_path -L/usr/local/lib/ -lprotobuf %odla_link -lodla_eigen -o %t_eigen.exe -Wno-deprecated-declarations
// RUN: %t_eigen.exe 0.0001 0 eigen %data_path/test_split_equal_parts_1d | FileCheck %s
// CHECK: Result Pass
// clang-format on
// XFAIL: *
#include "test_split_equal_parts_1d_eigen.cc.tmp.main.cc.in"
