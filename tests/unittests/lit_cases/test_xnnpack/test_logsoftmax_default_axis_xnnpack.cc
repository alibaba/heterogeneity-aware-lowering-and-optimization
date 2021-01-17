//===-test_logsoftmax_default_axis_xnnpack.cc-----------------------------------------------------------===//
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
// Testing CXX Code Gen using ODLA API on xnnpack
// RUN: %halo_compiler -target cxx -o %data_path/test_logsoftmax_default_axis/test_data_set_0/output_0.cc -x onnx -emit-data-as-c %data_path/test_logsoftmax_default_axis/test_data_set_0/output_0.pb
// RUN: %halo_compiler -target cxx -o %data_path/test_logsoftmax_default_axis/test_data_set_0/input_0.cc -x onnx -emit-data-as-c %data_path/test_logsoftmax_default_axis/test_data_set_0/input_0.pb
// RUN: %halo_compiler -target cxx -batch-size 1 %halo_compile_flags %data_path/test_logsoftmax_default_axis/model.onnx -o %t.cc
// RUN: %cxx -c -fPIC -o %t.o %t.cc -I%odla_path/include
// RUN: %cxx -g %s %t.o %t.bin -I%T -I%odla_path/include -I%unittests_path -I%data_path/test_logsoftmax_default_axis/test_data_set_0 %odla_link %device_link -lodla_xnnpack -o %t_xnnpack.exe -Wno-deprecated-declarations
// RUN: %t_xnnpack.exe 0.0001 0 xnnpack %data_path/test_logsoftmax_default_axis | FileCheck %s
// CHECK: Result Pass
// clang-format on
// XFAIL: *
#include "test_logsoftmax_default_axis_xnnpack.cc.tmp.main.cc.in"
