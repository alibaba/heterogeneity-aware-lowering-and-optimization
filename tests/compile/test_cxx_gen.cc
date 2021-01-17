//===- test_cxx_gen.cc ----------------------------------------------------===//
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

// REQUIRES: odla_dnnl

// RUN: %cxx %s -DCG_TEST -o %t %flags %include %link
// RUN: %t > %t.gen.cc
// RUN: cat %t.gen.cc | FileCheck %s --check-prefix=GEN

// Runtime test (build and for for mkldnn)
// RUN: %cxx %s -DRUNTIME_TEST -I%odla_path/include -c -o %t.main.o
// RUN: %cxx %t.gen.cc -I%odla_path/include -c -o %t.gen.o

// RUN: %cxx %t.gen.o  %t.main.o %odla_link -lodla_dnnl -o %t.mkl_exe
// RUN: %t.mkl_exe 2>&1| FileCheck %s --check-prefix=EXECUTE

// GEN: include <ODLA/odla.h>

// GEN: extern const float w0[3];
// GEN: extern const float w1[3];

// GEN: static odla_computation Comp;

// GEN: void func(const float input[3], float out_add1[3]) {
// GEN:  func_init();
// GEN:  odla_BindToArgumentById((const odla_value_id)"input", input, Ctx);
// GEN:  odla_BindToOutputById((const odla_value_id)"add1", out_add1, Ctx);
// GEN:  odla_ExecuteComputation(Comp, Ctx, ODLA_COMPUTE_INFERENCE, nullptr);
// GEN: }


// EXECUTE: 6.000000
// EXECUTE: 9.000000
// EXECUTE: 12.000000

// clang-format on
#include "test_cxx_gen.in"