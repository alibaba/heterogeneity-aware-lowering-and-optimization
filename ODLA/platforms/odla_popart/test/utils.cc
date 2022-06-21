
//===- utils.cc ----------------------------------------------------===//
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
#include "utils.h"

using namespace std;

odla_status build_default_model() {
  /**
   *
   * Const(2)  input
   *     \      /
   *        Mul     Const(6)
   *         |       |
   *          \     /
   *            Add
   *
   */

  std::vector<float> c2 = {2};
  std::vector<float> c6 = {6};
  auto Input = odla_CreateArgument({ODLA_FLOAT32, {.size = 1, .dims = {1}}},
                                   (const odla_value_id)("Input"));
  auto const_2 =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 1, .dims = {1}}}, c2.data(),
                          (const odla_value_id) "Mul_const");
  auto const_6 =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 1, .dims = {1}}}, c6.data(),
                          (const odla_value_id) "Add_const");
  auto Mul = odla_Mul(const_2, Input, (const odla_value_id) "Mul");
  auto Add = odla_Add(const_6, Mul, (const odla_value_id) "Add");
  odla_SetValueAsOutput(Add);
}

void set_computationItem(odla_computation comp, bool is_use_cpu, int ipu_nums,
                         int batches_per_step, bool enable_engine,
                         std::string cache_dir) {
  odla_SetComputationItem(comp, ODLA_USE_SIM_MODE,
                          (odla_item_value)&is_use_cpu);
  odla_SetComputationItem(comp, ODLA_PROCESSOR_NUM, (odla_item_value)&ipu_nums);
  odla_SetComputationItem(comp, ODLA_BATCHES_PER_STEP,
                          (odla_item_value)&batches_per_step);
  odla_SetComputationItem(comp, ODLA_ENABLE_ENGINE_CACHE,
                          (odla_item_value)&enable_engine);
  odla_SetComputationItem(comp, ODLA_CACHE_DIR,
                          (odla_item_value)cache_dir.c_str());
}

void test_bind_funciton_multithread(float* in, float* out) {
  odla_context ctx_multithread;
  odla_CreateContext(&ctx_multithread), ODLA_SUCCESS;
  odla_BindToArgumentById((const odla_value_id) "Input", in, ctx_multithread);
  odla_BindToOutputById((const odla_value_id) "Add", out, ctx_multithread);
  odla_DestroyContext(ctx_multithread);
}

void execute_multithread(odla_computation comp, float* in, float* out) {
  odla_context ctx_multithread;
  odla_CreateContext(&ctx_multithread);

  odla_BindToArgumentById((const odla_value_id) "Input", in, ctx_multithread);
  odla_BindToOutputById((const odla_value_id) "Add", out, ctx_multithread);
  odla_ExecuteComputation(comp, ctx_multithread, ODLA_COMPUTE_INFERENCE,
                          nullptr);

  odla_DestroyContext(ctx_multithread);
}

json default_json() {
  // Create a json object & fill with default value
  json jsonfile;
  jsonfile["amp"] = 0.6;
  jsonfile["sdk_version"] = popart::core::packageHash();
  jsonfile["version"] = std::string("1.0.0");
  jsonfile["batches_per_step"] = 1;
  jsonfile["ipu_num"] = 1;
  jsonfile["save_model"] = false;
  jsonfile["save_model_path"] = std::string("odla_popart_saved.onnx");
  jsonfile["load_onnx"] = false;
  jsonfile["load_onnx_path"] = std::string("not_set.onnx");
  jsonfile["execution_mode"] = std::string("sequence");
  jsonfile["queue_type"] = std::string("LockFreeQueue");
  jsonfile["queue_capacity"] = 1024 * 1024;
  jsonfile["debug"] = false;

  std::ofstream file("/tmp/tmp.json");
  file << jsonfile;
  return jsonfile;
}

void call_function(float param) {
  popart::logging::info({}, "call back function, parameter value", param);
}