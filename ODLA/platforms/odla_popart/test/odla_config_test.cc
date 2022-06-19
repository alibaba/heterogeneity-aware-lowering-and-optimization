//===- odla_config_test.cc
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

#include <array>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include "ODLA/odla_common.h"
#include "common.h"
#include "doctest.h"
#include "json.hpp"
#include "popart_config.h"
#include "utils.h"

using json = nlohmann::json;

TEST_CASE("TestConfig") {
  SUBCASE("TestCompExportCache") {
    json config_json = default_json();
    config_json["amp"] = 0.222;

    std::ofstream file("./test.json");
    file << config_json;
    file.close(); // must close before call compile_and_export

    PopartConfig::instance()->set_cache_path("./test.popart");
    PopartConfig::instance()->parse_from_json(config_json);

    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));

    build_default_model();
    set_computationItem(comp, false, 1);

    odla_context ctx;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateContext(&ctx));

    float in = 1.f;
    float out = 0.f;
    odla_BindToArgumentById((const odla_value_id) "Input", &in, ctx);
    odla_BindToOutputById((const odla_value_id) "Add", &out, ctx);

    CHECK_EQ(ODLA_SUCCESS, comp->compile_and_export());

    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
  }

  SUBCASE("TestCompLoadCache") {
    // cache file is not exist
    {
      PopartConfig::instance()->set_cache_path("./wrong_path");
      CHECK_EQ(PopartConfig::instance()->extract_config_from_cache(),
               ODLA_FAILURE);
    }

    // use cache (test.json)
    {
      PopartConfig::instance()->set_cache_path("./test.popart");
      CHECK_EQ(PopartConfig::instance()->extract_config_from_cache(),
               ODLA_SUCCESS);
      CHECK_EQ(PopartConfig::instance()->amp(), 0.222f);
      CHECK_EQ(PopartConfig::instance()->ipu_num(), 1);
      CHECK_EQ(PopartConfig::instance()->execution_mode(),
               ExecutionMode::SEQUENCE);
    }
    // CHECK_EQ(PopartConfig::instance()->load_from_file("./test.popart"),
    // ODLA_SUCCESS);
  }

  SUBCASE("TestConfigGetFunction") {
    // Check parse_from_json
    json config_json = default_json();
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
    auto pci = PopartConfig::instance();

    CHECK_EQ(true, pci->inited());
    CHECK_EQ(0.6f, pci->amp());
    CHECK_EQ(1, pci->batches_per_step());
    CHECK_EQ(1, pci->ipu_num());
    CHECK_EQ(false, pci->save_model());
    CHECK_EQ(std::string("1.0.0"), pci->version());
    CHECK_EQ(std::string("odla_popart_saved.onnx"), pci->save_model_path());
    CHECK_EQ(false, pci->load_onnx());
    CHECK_EQ(std::string("not_set.onnx"), pci->load_onnx_path());
    CHECK_EQ(ExecutionMode::SEQUENCE, pci->execution_mode());
    CHECK_EQ(std::string("LockFreeQueue"), pci->queue_type());
    CHECK_EQ(1024 * 1024, pci->queue_capacity());
    CHECK_EQ(false, pci->debug());

    int64_t ipu_idx = -1, pipeline_stage = -1;
    pci->get_pipeline_setting("Input", ipu_idx, pipeline_stage);
    CHECK_EQ(0, ipu_idx);
    CHECK_EQ(0, pipeline_stage);

    pci->get_pipeline_setting("Mul", ipu_idx, pipeline_stage);
    CHECK_EQ(1, ipu_idx);
    CHECK_EQ(1, pipeline_stage);
  }

  SUBCASE("TestUnexpectedSetting") {
    // test odla_SetComputationItem 1001, 1002
    {
      odla_computation comp;
      CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));

      // 1001 load cache directly, need set path of cache file
      std::string cache_dir = "./tmp";
      CHECK_EQ(odla_SetComputationItem(comp, (odla_item_type)1001,
                                       (odla_item_value)cache_dir.c_str()),
               ODLA_SUCCESS);

      // unsuport type 1002
      CHECK_EQ(odla_SetComputationItem(comp, (odla_item_type)1002, nullptr),
               ODLA_UNSUPPORTED_DATATYPE);
      CHECK_EQ(odla_DestroyComputation(comp), ODLA_SUCCESS);
    }

    // test odla_SetContextItem ODLA_ASYNC_CALLBACK_ARG, unexpected type
    {
      odla_computation comp;
      CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));

      odla_context ctx;
      odla_CreateContext(&ctx);
      odla_item_value _test;

      CHECK_EQ(odla_SetContextItem(ctx, ODLA_ASYNC_CALLBACK_ARG,
                                   (odla_item_value)&_test),
               ODLA_SUCCESS);
      CHECK_EQ(
          odla_SetContextItem(ctx, (odla_item_type)0, (odla_item_value)&_test),
          ODLA_UNSUPPORTED_DATATYPE);

      CHECK_EQ(odla_DestroyComputation(comp), ODLA_SUCCESS);
      CHECK_EQ(odla_DestroyContext(ctx), ODLA_SUCCESS);
    }
  }

  SUBCASE("TestMakeNDArrayWrapperType") {
    odla_void* val;
    std::vector<int64_t> shape(1, 1);
    auto ndarray = MakeNDArrayWrapper(val, popart::DataType::FLOAT16, shape);
    CHECK_EQ(popart::DataType::FLOAT16, ndarray.get()->dataType());

    ndarray = MakeNDArrayWrapper(val, popart::DataType::UINT32, shape);
    CHECK_EQ(popart::DataType::UINT32, ndarray.get()->dataType());

    ndarray = MakeNDArrayWrapper(val, popart::DataType::BOOL, shape);
    CHECK_EQ(popart::DataType::BOOL, ndarray.get()->dataType());

    ndarray = MakeNDArrayWrapper(val, popart::DataType::INT64, shape);
    CHECK_EQ(popart::DataType::INT64, ndarray.get()->dataType());

    ndarray = MakeNDArrayWrapper(val, popart::DataType::INT32, shape);
    CHECK_EQ(popart::DataType::INT32, ndarray.get()->dataType());

    ndarray = MakeNDArrayWrapper(val, popart::DataType::FLOAT, shape);
    CHECK_EQ(popart::DataType::FLOAT, ndarray.get()->dataType());
  }

  SUBCASE("TestConfigFunctionCallAfterInited") {
    json config_json = default_json();
    config_json["execution_mode"] = "parallel";
    config_json["queue_type"] = "ContextQueues";
    config_json["queue_capacity"] = 512 * 512;

    std::string path("./TestConfigFunctionCallAfterInited.json");
    std::ofstream file(path);
    file << config_json;
    file.close();

    CHECK_EQ(ODLA_SUCCESS, PopartConfig::instance()->load_config(path.c_str()));
    CHECK_EQ(true, PopartConfig::instance()->inited());
    CHECK_EQ(std::string("ContextQueues"),
             PopartConfig::instance()->queue_type());
    CHECK_EQ(512 * 512, PopartConfig::instance()->queue_capacity());
    CHECK_EQ(ExecutionMode::PARALLEL,
             PopartConfig::instance()->execution_mode());

    CHECK_NOTHROW_MESSAGE(
        PopartConfig::instance()->parse_from_json(config_json),
        "config already inited");
    CHECK_NOTHROW_MESSAGE(PopartConfig::instance()->load_config(path.c_str()),
                          "config already inited");
  }

  SUBCASE("TestDefualtConfig") {
    auto pci = PopartConfig::instance();
    pci->use_default();
    CHECK_EQ(true, pci->inited());
    CHECK_EQ(0.6f, pci->amp());
    CHECK_EQ(1, pci->batches_per_step());
    CHECK_EQ(1, pci->ipu_num());
    CHECK_EQ(false, pci->save_model());
    CHECK_EQ(std::string("1.0.0"), pci->version());
    CHECK_EQ(std::string("odla_popart_saved.onnx"), pci->save_model_path());
    CHECK_EQ(false, pci->load_onnx());
    CHECK_EQ(std::string("not_set.onnx"), pci->load_onnx_path());
    CHECK_EQ(ExecutionMode::SEQUENCE, pci->execution_mode());
    CHECK_EQ(std::string("LockFreeQueue"), pci->queue_type());
    CHECK_EQ(1024 * 1024, pci->queue_capacity());
    CHECK_EQ(false, pci->debug());
  }

  SUBCASE("TestCompEnviroment") {
    // POPART_LOG_LEVEL is not set, should setLogLevel warn
    odla_computation comp;
    unsetenv("POPART_LOG_LEVEL");
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    CHECK_EQ(popart::logging::Level::Warn,
             popart::logging::getLogLevel(popart::logging::Module::popart));
    odla_DestroyComputation(comp);

    // set POPLAR_ENGINE_OPTIONS, should message out
    setenv("POPLAR_ENGINE_OPTIONS",
           "\'{\"autoReport.all\":\"true\", "
           "\"autoReport.directory\":\"profile_out\"}\'",
           1);
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
  }

  SUBCASE("TestCompEnviroment2") {
    // have injector but not set POPART_LOG_LEVEL, should set to info
    json inject_error;
    std::ofstream file("/tmp/temp_error_injector.json");
    inject_error["POPLAR_ENGINE_OPTIONS"] =
        "{\"debug.simulateErrors\":\"MEMORY_ERROR@ALL:vertexName:popops__"
        "BroadcastScalar1DSupervisor___popops__expr__BinaryOpType__SUBTRACT_"
        "float\"}";
    file << inject_error;
    file.close();

    unsetenv("POPART_LOG_LEVEL");
    odla_computation comp;
    odla_context ctx;

    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    CHECK_EQ(popart::logging::Level::Info,
             popart::logging::getLogLevel(popart::logging::Module::popart));

    remove("/tmp/temp_error_injector.json");
    odla_DestroyComputation(comp);
  }

  SUBCASE("TestGetOdlaPopartType") {
    CHECK_EQ(popart::DataType::FLOAT,
             GetPopartType(
                 (odla_value_type){ODLA_FLOAT32, {.size = 1, .dims = {1}}}));
    CHECK_EQ(popart::DataType::FLOAT16,
             GetPopartType(
                 (odla_value_type){ODLA_FLOAT16, {.size = 1, .dims = {1}}}));
    CHECK_EQ(
        popart::DataType::INT32,
        GetPopartType((odla_value_type){ODLA_INT32, {.size = 1, .dims = {1}}}));
    CHECK_EQ(
        popart::DataType::INT64,
        GetPopartType((odla_value_type){ODLA_INT64, {.size = 1, .dims = {1}}}));
    CHECK_EQ(popart::DataType::UINT32,
             GetPopartType(
                 (odla_value_type){ODLA_UINT32, {.size = 1, .dims = {1}}}));
    CHECK_EQ(popart::DataType::UINT64,
             GetPopartType(
                 (odla_value_type){ODLA_UINT64, {.size = 1, .dims = {1}}}));
    CHECK_EQ(popart::DataType::BOOL, GetPopartType((odla_value_type){
                                         ODLA_BOOL, {.size = 1, .dims = {1}}}));

    CHECK_EQ(ODLA_BOOL, GetOdlaType(popart::DataType::BOOL));
    CHECK_EQ(ODLA_FLOAT32, GetOdlaType(popart::DataType::FLOAT));
    CHECK_EQ(ODLA_FLOAT16, GetOdlaType(popart::DataType::FLOAT16));
    CHECK_EQ(ODLA_INT32, GetOdlaType(popart::DataType::INT32));
    CHECK_EQ(ODLA_INT64, GetOdlaType(popart::DataType::INT64));
    CHECK_EQ(ODLA_UINT32, GetOdlaType(popart::DataType::UINT32));
    CHECK_EQ(ODLA_UINT64, GetOdlaType(popart::DataType::UINT64));
  }
}
