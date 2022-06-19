//===- utils.h ----------------------------------------------------===//
//
// Copyright (C) 2022 Alibaba Group Holding Limited.
// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
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
#include <ODLA/odla.h>
#include <stdio.h>

#include <array>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include "ODLA/odla_common.h"
#include "odla_popart.h"
#include "popart_config.h"

#define TOLLERANCE 0.001f

odla_status build_default_model();

void set_computationItem(odla_computation comp, bool is_use_cpu = true,
                         int ipu_nums = 1, int batches_per_step = 1,
                         bool enable_engine = false,
                         std::string cache_dir = "/tmp/");

void test_bind_funciton_multithread(float* in, float* out);
void execute_multithread(odla_computation comp, float* in, float* out);

json default_json();
void call_function(float param);