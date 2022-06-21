//===- odla_compute_test.cc
//----------------------------------------------------===//
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
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <ODLA/odla.h>
#include <stdio.h>

#include <array>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include "ODLA/odla_common.h"
#include "doctest.h"
#include "odla_popart.h"
#include "popart_config.h"
#include "utils.h"

typedef unsigned short uint16_t;

using namespace std;

TEST_CASE("TestBaseCompFunction") {
  SUBCASE("TestCompInitDestroy") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem(comp);

    odla_context ctx;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateContext(&ctx));
    int wrong_addr;
    CHECK_EQ(odla_CreateComputation(&comp), ODLA_SUCCESS);
    // CHECK_EQ(odla_DestroyComputation(nullptr), ODLA_FAILURE); //todo 1:
    // nullptr wrong addr protest
    // CHECK_EQ(odla_DestroyComputation((odla_computation)&wrong_addr),
    // ODLA_FAILURE);

    CHECK_EQ(odla_CreateContext(&ctx), ODLA_SUCCESS);
    // CHECK_EQ(odla_DestroyContext(nullptr), ODLA_FAILURE); //todo 1
    // CHECK_EQ(odla_DestroyContext((odla_context)&wrong_addr), ODLA_FAILURE);

    odla_item_value _test;
    // CHECK_EQ(odla_SetComputationItem(nullptr, ODLA_USE_SIM_MODE,
    // (odla_item_value)&_test), ODLA_FAILURE); // todo: unvaild value, should
    // be recognized failure
    // CHECK_EQ(odla_SetComputationItem((odla_computation)&wrong_addr,
    // ODLA_USE_SIM_MODE, (odla_item_value)&_test), ODLA_FAILURE); //todo 1
    CHECK_EQ(odla_SetComputationItem(comp, ODLA_USE_SIM_MODE,
                                     (odla_item_value)&_test),
             ODLA_SUCCESS);
    CHECK_EQ(odla_SetComputationItem(comp, ODLA_LOAD_ENGINE_MODE,
                                     (odla_item_value)&_test),
             ODLA_UNSUPPORTED_DATATYPE);

    // CHECK_EQ(odla_SetContextItem(nullptr, ODLA_ASYNC_CALLBACK_FUNC,
    // (odla_item_value)&_test), ODLA_INVALID_PARAM); //todo 1
    // CHECK_EQ(odla_SetContextItem((odla_context)&wrong_addr,
    // ODLA_ASYNC_CALLBACK_FUNC, (odla_item_value)&_test), ODLA_INVALID_PARAM);
    CHECK_EQ(odla_SetContextItem(ctx, ODLA_ASYNC_CALLBACK_FUNC,
                                 (odla_item_value)&_test),
             ODLA_SUCCESS);

    CHECK_EQ(odla_DestroyComputation(comp), ODLA_SUCCESS);
    CHECK_EQ(odla_DestroyContext(ctx), ODLA_SUCCESS);
  }

  SUBCASE("TestCtxNumFunciton") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem(comp);

    odla_context ctx;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateContext(&ctx));

    int wrong_addr;
    int data[5] = {0};
    odla_uint32 _num, _id;
    odla_value _ov;

    auto _input1 = odla_CreateArgument({ODLA_FLOAT32, {.size = 1, .dims = {1}}},
                                       (const odla_value_id)("_input1"));
    auto _input2 = odla_CreateArgument({ODLA_FLOAT32, {.size = 1, .dims = {1}}},
                                       (const odla_value_id)("_input2"));

    auto _constance =
        odla_CreateConstant({ODLA_FLOAT32, {.size = 1, .dims = {1}}}, &data,
                            (const odla_value_id) "_constance");

    auto _constance1 =
        odla_CreateConstant({ODLA_FLOAT32, {.size = 8, .dims = {1}}},
                            (int*)0x11, (const odla_value_id) "_constance");

    CHECK_EQ(odla_GetNumOfArgsFromComputation(comp, &_num), ODLA_SUCCESS);
    // CHECK_EQ(odla_GetNumOfArgsFromComputation(nullptr, &_num), ODLA_FAILURE);
    // // todo 1
    // CHECK_EQ(odla_GetNumOfArgsFromComputation((odla_computation)&wrong_addr,
    // &_num), ODLA_FAILURE);
    CHECK_EQ(_num, 2);

    CHECK_EQ(odla_GetArgFromComputationByIdx(comp, 0, &_ov), ODLA_SUCCESS);
    CHECK_EQ(odla_GetArgFromComputationByIdx(comp, 2, &_ov),
             ODLA_INVALID_PARAM);
    // CHECK_EQ(odla_GetArgFromComputationByIdx(nullptr, 0, &_ov),
    // ODLA_FAILURE); //todo 1
    // CHECK_EQ(odla_GetArgFromComputationByIdx((odla_computation)&wrong_addr,
    // 0, &_ov), ODLA_FAILURE);

    CHECK_EQ(odla_SetValueAsOutput(_input1), ODLA_SUCCESS);
    // CHECK_EQ(odla_SetValueAsOutput(_input1), ODLA_FAILURE); //todo: double
    // set, should be failed
    CHECK_EQ(odla_SetValueAsOutput(_input2), ODLA_SUCCESS);

    odla_values _ovs = {2, {_input1, _input2}};
    CHECK_EQ(odla_SetValuesAsOutput(_ovs), ODLA_SUCCESS); // todo: duplicate
    // set, should be failed
    CHECK_EQ(odla_GetNumOfOutputsFromComputation(comp, &_num), ODLA_SUCCESS);
    // CHECK_EQ(_num, 2); //todo: duplicate set, should be failed

    CHECK_EQ(odla_GetOutputFromComputationByIdx(comp, 0, &_ov), ODLA_SUCCESS);
    CHECK_EQ(odla_GetOutputFromComputationByIdx(comp, 4, &_ov),
             ODLA_INVALID_PARAM);
    // CHECK_EQ(odla_GetOutputFromComputationByIdx(nullptr, 0, &_ov),
    // ODLA_FAILURE); //todo 1
    // CHECK_EQ(odla_GetOutputFromComputationByIdx((odla_computation)&wrong_addr,
    // 0, &_ov), ODLA_FAILURE);

    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
  }

  SUBCASE("TestCtxBindInputOutput") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));

    set_computationItem(comp);
    build_default_model();

    odla_context ctx;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateContext(&ctx));

    int wrong_addr;
    float in = 1.f;
    float out = 1.f;

    // CHECK_EQ(odla_BindToArgumentById((const odla_value_id) "Input", &in,
    // nullptr), ODLA_FAILURE); //todo 1 CHECK_EQ(odla_BindToArgumentById((const
    // odla_value_id) "Input", &in, (odla_context)&wrong_addr), ODLA_FAILURE);
    CHECK_EQ(odla_BindToArgumentById((const odla_value_id) "Input", &in, ctx),
             ODLA_SUCCESS);
    // CHECK_EQ(odla_BindToArgumentById((const odla_value_id) "Input", &in,
    // ctx), ODLA_FAILURE); // todo duplicate bind, should be recognized failure
    // CHECK_EQ(odla_BindToArgumentById((const odla_value_id) "Input", nullptr,
    // ctx), ODLA_FAILURE); //todo 1

    // CHECK_EQ(odla_BindToOutputById((const odla_value_id) "Add", &out,
    // nullptr), ODLA_FAILURE); CHECK_EQ(odla_BindToOutputById((const
    // odla_value_id) "Add", &out, (odla_context)&wrong_addr), ODLA_FAILURE);
    // //todo 1
    CHECK_EQ(odla_BindToOutputById((const odla_value_id) "Add", &out, ctx),
             ODLA_SUCCESS);
    // CHECK_EQ(odla_BindToOutputById((const odla_value_id) "Add", &out, ctx),
    // ODLA_FAILURE); //todo duplicate bind, should be recognized failure
    // CHECK_EQ(odla_BindToOutputById((const odla_value_id) "Add", nullptr,
    // ctx), ODLA_FAILURE);

    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
  }

  SUBCASE("TestCtxBindInputOutputPThread") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));

    build_default_model();
    set_computationItem(comp);

    std::thread threads[5];
    float in[5], out[5];

    for (int i = 0; i < 5; i++) {
      threads[i] = std::thread(test_bind_funciton_multithread, &in[i], &out[i]);
    }
    for (auto& t : threads) {
      t.join();
    }
    CHECK_EQ(ODLA_SUCCESS, odla_DestroyComputation(comp));
  }

  SUBCASE("TestGetValueType") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));

    set_computationItem(comp);
    build_default_model();

    odla_context ctx;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateContext(&ctx));

    auto _ov1 = odla_CreateArgument({ODLA_FLOAT32, {.size = 1, .dims = {1}}},
                                    (const odla_value_id)("_ov1"));
    odla_value _ov2;
    odla_value_type _ovt;
    CHECK_EQ(odla_GetValueType(_ov1, &_ovt), ODLA_SUCCESS);
    CHECK_EQ(_ovt.element_type, ODLA_FLOAT32);
    CHECK_EQ(_ovt.shape.size, 1);

    // CHECK_EQ(odla_GetValueType(_ov2, &_ovt), ODLA_FAILURE); // todo unvaild
    // value

    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
  }

  SUBCASE("TestContextQueue") {
    json config_json = default_json();
    config_json["queue_type"] = std::string("ContextQueues");
    config_json["ipu_num"] = 2;
    config_json["execution_mode"] = std::string("pipeline");
    json pipeline_json;
    std::vector<int> vec1 = {0, 0};
    std::vector<int> vec2 = {1, 1};
    pipeline_json["Input"] = vec1;
    pipeline_json["Mul"] = vec2;
    pipeline_json["Mul_const"] = vec2;
    pipeline_json["Add"] = vec2;
    pipeline_json["Add_const"] = vec2;
    config_json["pipeline"] = pipeline_json;

    PopartConfig::instance()->parse_from_json(config_json);

    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));

    build_default_model();
    set_computationItem(comp, false, 2);

    odla_context ctx;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateContext(&ctx));

    float in = 1.f;
    float out = 0.f;

    odla_BindToArgumentById((const odla_value_id) "Input", &in, ctx);
    odla_BindToOutputById((const odla_value_id) "Add", &out, ctx);

    CHECK_EQ(
        odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr),
        ODLA_SUCCESS);

    CHECK_EQ(out, 8.f);

    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
  }

  SUBCASE("TestPipelineAsync") {
    json config_json = default_json();
    config_json["execution_mode"] = "pipeline_async";
    config_json["ipu_num"] = 2;
    json pipeline_json;
    std::vector<int> vec1 = {0, 0};
    std::vector<int> vec2 = {1, 1};
    pipeline_json["Input"] = vec1;
    pipeline_json["Mul"] = vec2;
    pipeline_json["Mul_const"] = vec2;
    pipeline_json["Add"] = vec2;
    pipeline_json["Add_const"] = vec2;
    config_json["pipeline"] = pipeline_json;

    PopartConfig::instance()->parse_from_json(config_json);

    odla_computation comp;
    odla_CreateComputation(&comp);

    build_default_model();
    set_computationItem(comp, true, 2);

    odla_context ctx;
    odla_CreateContext(&ctx);

    float in = 1.f;
    float out = 0.f;
    odla_BindToArgumentById((const odla_value_id) "Input", &in, ctx);
    odla_BindToOutputById((const odla_value_id) "Add", &out, ctx);
    float parameter = 1;
    odla_SetContextItem(ctx, ODLA_ASYNC_CALLBACK_FUNC,
                        (odla_item_value)&call_function);
    odla_SetContextItem(ctx, ODLA_ASYNC_CALLBACK_ARG,
                        (odla_item_value)&parameter);
    CHECK_EQ(ODLA_SUCCESS, odla_AsyncExecuteComputation(
                               comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr));
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
  }

  SUBCASE("TestSequenceExecuteModelPthread") {
    float in[3] = {1.f, 1.f, 1.f};
    float out[3] = {0.f};

    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));

    build_default_model();
    set_computationItem(comp);

    std::thread t1(execute_multithread, comp, &in[0], &out[0]);
    std::thread t2(execute_multithread, comp, &in[1], &out[1]);
    std::thread t3(execute_multithread, comp, &in[2], &out[2]);

    t1.join();
    t2.join();
    t3.join();

    CHECK_EQ(out[0], 8.f);
    CHECK_EQ(out[1], 8.f);
    CHECK_EQ(out[2], 8.f);

    odla_DestroyComputation(comp);
  }

  SUBCASE("TestPipelineExecuteModelPthread") {
    json config_json = default_json();
    config_json["ipu_num"] = 2;
    config_json["execution_mode"] = std::string("pipeline");
    json pipeline_json;
    std::vector<int> vec1 = {0, 0};
    std::vector<int> vec2 = {1, 1};
    pipeline_json["Input"] = vec1;
    pipeline_json["Mul"] = vec2;
    pipeline_json["Mul_const"] = vec2;
    pipeline_json["Add"] = vec2;
    pipeline_json["Add_const"] = vec2;
    config_json["pipeline"] = pipeline_json;

    PopartConfig::instance()->parse_from_json(config_json);

    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));

    build_default_model();
    set_computationItem(comp, false, 2);

    odla_context ctx;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateContext(&ctx));

    float in = 1.f;
    float out = 0.f;

    odla_BindToArgumentById((const odla_value_id) "Input", &in, ctx);
    odla_BindToOutputById((const odla_value_id) "Add", &out, ctx);

    CHECK_EQ(
        odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr),
        ODLA_SUCCESS);

    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
  }

  SUBCASE("TestSaveOnnx") {
    json config_json = default_json();
    config_json["save_model"] = true;
    config_json["save_model_path"] = std::string("./model.onnx");
    PopartConfig::instance()->parse_from_json(config_json);

    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));

    build_default_model();
    set_computationItem(comp);

    odla_context ctx;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateContext(&ctx));

    float in = 1.f;
    float out = 0.f;
    odla_BindToArgumentById((const odla_value_id) "Input", &in, ctx);
    odla_BindToOutputById((const odla_value_id) "Add", &out, ctx);

    CHECK_EQ(
        odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr),
        ODLA_SUCCESS);
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
    std::fstream model("./model.onnx");
    CHECK_EQ(true, model.good());
  }

  SUBCASE("TestLoadOnnx") {
   json config_json = default_json();
    config_json["load_onnx"] = true;
    config_json["load_onnx_path"] = std::string("./model.onnx");
    PopartConfig::instance()->parse_from_json(config_json);

    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));

    build_default_model();
    set_computationItem(comp);

    odla_context ctx;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateContext(&ctx));

    float in = 1.f;
    float out = 0.f;
    odla_BindToArgumentById((const odla_value_id) "Input", &in, ctx);
    odla_BindToOutputById((const odla_value_id) "Add", &out, ctx);

    CHECK_EQ(
        odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr),
        ODLA_SUCCESS);
    CHECK_EQ(out, 8.f); //todo need to change verify way.
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
  }

  SUBCASE("TestInjectError") {
    // generate json
    json config_json = default_json();
    PopartConfig::instance()->parse_from_json(config_json);

    json inject_error;
    std::ofstream file("/tmp/temp_error_injector.json");
    inject_error["POPLAR_ENGINE_OPTIONS"] =
        "{\"debug.simulateErrors\":\"MEMORY_ERROR@ALL:vertexName:popops__"
        "BroadcastScalar1DSupervisor___popops__expr__BinaryOpType__SUBTRACT_"
        "float\"}";
    file << inject_error;
    file.close();

    std::string path = "./test";
    PopartConfig::instance()->set_cache_path(path);

    odla_computation comp;
    odla_context ctx;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));

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
    auto Sub = odla_Sub(const_6, Mul, (const odla_value_id) "Sub");
    odla_SetValueAsOutput(Sub);

    odla_CreateContext(&ctx);
    set_computationItem(comp, false, 1);

    odla_context ctx_multithread;
    odla_CreateContext(&ctx_multithread);

    float in = 1.f, out = 0.f;
    odla_BindToArgumentById((const odla_value_id) "Input", &in,
                            ctx_multithread);
    odla_BindToOutputById((const odla_value_id) "Sub", &out, ctx_multithread);
    CHECK_EQ(odla_ExecuteComputation(comp, ctx_multithread,
                                     ODLA_COMPUTE_INFERENCE, nullptr),
             ODLA_RECOVERABLE_ERR);

    remove("/tmp/temp_error_injector.json");
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
  }

  SUBCASE("TestIPUNumNotEqualFailure") {
    json config_json = default_json();
    config_json["ipu_num"] = 2;
    config_json["execution_mode"] = std::string("pipeline_async");
    json pipeline_json;
    std::vector<int> vec1 = {0, 0};
    std::vector<int> vec2 = {1, 1};
    pipeline_json["Input"] = vec1;
    pipeline_json["Mul"] = vec2;
    pipeline_json["Mul_const"] = vec2;
    pipeline_json["Add"] = vec2;
    pipeline_json["Add_const"] = vec2;
    config_json["pipeline"] = pipeline_json;
    PopartConfig::instance()->parse_from_json(config_json);

    float in = 1.f;
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    build_default_model();
    set_computationItem(comp, true, 1);

    odla_context ctx;
    odla_CreateContext(&ctx);

    // Info: number of ipus in pipeline configuration:2 must same with options:
    // 1 Info: init computation item in CreateContext failed Info: Can't notify
    // the failure status: 14 as null callback func:0 or arg:0
    CHECK_EQ(ODLA_FAILURE,
             odla_BindToArgumentById((const odla_value_id) "Input", &in, ctx));
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
  }
}
