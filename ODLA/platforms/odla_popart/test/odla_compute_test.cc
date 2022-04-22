//===- Halo Compiler Generated File --------------------------------===//
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <ODLA/odla.h>
#include <stdio.h>

#include <array>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include "odla_popart.h"
#include "doctest.h"
#include "popart_config.h"
#include "ODLA/odla_common.h"

typedef unsigned short uint16_t;




using namespace std;

odla_status model_helper() 
{
  std::vector<float> py = {2};
  std::vector<float> pz = {3};
  auto Input =
    odla_CreateArgument({ODLA_FLOAT32, {.size = 1, .dims = {1}}},
                        (const odla_value_id)("Input"));
  auto py_ = odla_CreateConstant({ODLA_FLOAT32, {.size = 1, .dims = {1}}},
                                py.data(), (const odla_value_id) "Mul_const");
  auto pz_ = odla_CreateConstant({ODLA_FLOAT32, {.size = 1, .dims = {1}}},
                                pz.data(), (const odla_value_id) "Add_const");
  auto Mul = odla_Mul(py_, Input, (const odla_value_id) "Mul");
  auto Add = odla_Add(pz_, Mul, (const odla_value_id) "Add");
  odla_SetValueAsOutput(Add);
}

void set_computationItem(odla_computation comp, int ipu_nums) {
  bool use_ipu_model = 1;
  int ipu_num = ipu_nums;
  int batches_per_step = 1;
  odla_SetComputationItem(comp, ODLA_USE_SIM_MODE,
                          (odla_item_value)&use_ipu_model);
  odla_SetComputationItem(comp, ODLA_PROCESSOR_NUM, (odla_item_value)&ipu_num);
  odla_SetComputationItem(comp, ODLA_BATCHES_PER_STEP,
                          (odla_item_value)&batches_per_step);
}

void test_bind_funciton_multithread(float* in, float* out) 
{
  odla_context ctx_multithread;
  CHECK_EQ(odla_CreateContext(&ctx_multithread), ODLA_SUCCESS);
  CHECK_EQ(odla_BindToArgumentById((const odla_value_id)"Input", in, ctx_multithread), ODLA_SUCCESS);
  CHECK_EQ(odla_BindToOutputById((const odla_value_id) "Add", out, ctx_multithread), ODLA_SUCCESS);
  odla_DestroyContext(ctx_multithread);

}

void execute_multithread(odla_computation comp, float* in, float* out)
{
    odla_context ctx_multithread;
    odla_CreateContext(&ctx_multithread);

    odla_BindToArgumentById((const odla_value_id) "Input", in, ctx_multithread);
    odla_BindToOutputById((const odla_value_id) "Add", out, ctx_multithread);
    odla_ExecuteComputation(comp, ctx_multithread, ODLA_COMPUTE_INFERENCE, nullptr);

    odla_DestroyContext(ctx_multithread);
}

 json default_json(
        float amp = 0.6,
        const std::string& sdk_version = popart::core::packageHash(),
        const std::string& version = std::string("1.0.0"), int batches_per_step = 1,
        int ipu_num = 1, bool save_model = false,
        const std::string& save_model_path = std::string("odla_popart_saved.onnx"),
        bool load_onnx = false,
        const std::string& load_onnx_path = std::string("not_set.onnx"),
        const std::string& execution_mode = std::string("sequence"),
        const std::string& queue_type = std::string("LockFreeQueue"),
        int queue_capacity = 1024 * 1024, bool debug = false) {
      // Create a json object & fill with default value
      json jsonfile;
      jsonfile["amp"] = amp;
      jsonfile["sdk_version"] = sdk_version;
      jsonfile["version"] = version;
      jsonfile["batches_per_step"] = batches_per_step;
      jsonfile["ipu_num"] = ipu_num;
      jsonfile["save_model"] = save_model;
      jsonfile["save_model_path"] = save_model_path;
      jsonfile["load_onnx"] = false;
      jsonfile["load_onnx_path"] = load_onnx_path;
      jsonfile["execution_mode"] = execution_mode;
      jsonfile["queue_type"] = queue_type;
      jsonfile["queue_capacity"] = queue_capacity;
      jsonfile["debug"] = debug;

      json pipeline;
      std::vector<int> vec1, vec2;
      vec1.push_back(0);
      vec1.push_back(0);
      vec2.push_back(1);
      vec2.push_back(1);
      pipeline["Input"] = vec1;
      pipeline["Mul"] = vec1;
      pipeline["Mul_const"] = vec1;
      pipeline["Add"] = vec1;
      pipeline["Add_const"] = vec1;

      jsonfile["pipeline"] = pipeline;

      std::ofstream file("/tmp/tmp.json");
      file << jsonfile;
      return jsonfile;
    }


TEST_CASE("testing base interface") 
  {
    
    SUBCASE("test base function") 
    {
      odla_computation comp;
      odla_context ctx;
      CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
      odla_CreateContext(&ctx);
      set_computationItem(comp, 1);

      int wrong_addr;
      CHECK_EQ(odla_CreateComputation(&comp), ODLA_SUCCESS);
      // CHECK_EQ(odla_DestroyComputation(nullptr), ODLA_FAILURE); //todo 1: nullptr wrong addr protest 
      // CHECK_EQ(odla_DestroyComputation((odla_computation)&wrong_addr), ODLA_FAILURE);

      CHECK_EQ(odla_CreateContext(&ctx), ODLA_SUCCESS);
      // CHECK_EQ(odla_DestroyContext(nullptr), ODLA_FAILURE); //todo 1
      // CHECK_EQ(odla_DestroyContext((odla_context)&wrong_addr), ODLA_FAILURE);

      odla_item_value _test;
      // CHECK_EQ(odla_SetComputationItem(nullptr, ODLA_USE_SIM_MODE, (odla_item_value)&_test), ODLA_FAILURE); // todo: unvaild value, should be recognized failure
      // CHECK_EQ(odla_SetComputationItem((odla_computation)&wrong_addr, ODLA_USE_SIM_MODE, (odla_item_value)&_test), ODLA_FAILURE); //todo 1
      CHECK_EQ(odla_SetComputationItem(comp, ODLA_USE_SIM_MODE, (odla_item_value)&_test), ODLA_SUCCESS);
      CHECK_EQ(odla_SetComputationItem(comp, ODLA_LOAD_ENGINE_MODE, (odla_item_value)&_test), ODLA_UNSUPPORTED_DATATYPE);

      // CHECK_EQ(odla_SetContextItem(nullptr, ODLA_ASYNC_CALLBACK_FUNC, (odla_item_value)&_test), ODLA_INVALID_PARAM); //todo 1
      // CHECK_EQ(odla_SetContextItem((odla_context)&wrong_addr, ODLA_ASYNC_CALLBACK_FUNC, (odla_item_value)&_test), ODLA_INVALID_PARAM);
      CHECK_EQ(odla_SetContextItem(ctx, ODLA_ASYNC_CALLBACK_FUNC, (odla_item_value)&_test), ODLA_SUCCESS);

      CHECK_EQ(odla_DestroyComputation(comp), ODLA_SUCCESS);
      CHECK_EQ(odla_DestroyContext(ctx), ODLA_SUCCESS);

    }

    SUBCASE("test arg function") 
    {    
      
      odla_computation comp;
      odla_context ctx;
      CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
      odla_CreateContext(&ctx);
      set_computationItem(comp, 1);
      
      int wrong_addr;
      int data[5] = {0};
      odla_uint32 _num, _id;
      odla_value _ov;

      auto _input1 = odla_CreateArgument({ODLA_FLOAT32, {.size = 1, .dims = {1}}},
                          (const odla_value_id)("_input1"));
      auto _input2 = odla_CreateArgument({ODLA_FLOAT32, {.size = 1, .dims = {1}}},
                          (const odla_value_id)("_input2"));

      auto _constance = odla_CreateConstant({ODLA_FLOAT32, {.size = 1, .dims = {1}}}, &data,
                          (const odla_value_id) "_constance");

      auto _constance1 = odla_CreateConstant({ODLA_FLOAT32, {.size = 8, .dims = {1}}},
                          (int*)0x11, (const odla_value_id) "_constance");

      CHECK_EQ(odla_GetNumOfArgsFromComputation(comp, &_num), ODLA_SUCCESS);
      // CHECK_EQ(odla_GetNumOfArgsFromComputation(nullptr, &_num), ODLA_FAILURE); // todo 1
      // CHECK_EQ(odla_GetNumOfArgsFromComputation((odla_computation)&wrong_addr, &_num), ODLA_FAILURE);
      CHECK_EQ(_num, 2);

      CHECK_EQ(odla_GetArgFromComputationByIdx(comp, 0, &_ov), ODLA_SUCCESS);
      CHECK_EQ(odla_GetArgFromComputationByIdx(comp, 2, &_ov), ODLA_INVALID_PARAM);
      // CHECK_EQ(odla_GetArgFromComputationByIdx(nullptr, 0, &_ov), ODLA_FAILURE); //todo 1
      // CHECK_EQ(odla_GetArgFromComputationByIdx((odla_computation)&wrong_addr, 0, &_ov), ODLA_FAILURE);

      CHECK_EQ(odla_SetValueAsOutput(_input1), ODLA_SUCCESS);
      // CHECK_EQ(odla_SetValueAsOutput(_input1), ODLA_FAILURE); //todo: double set, should be failed
      CHECK_EQ(odla_SetValueAsOutput(_input2), ODLA_SUCCESS);

      odla_values _ovs = {2, {_input1, _input2}};
      // CHECK_EQ(odla_SetValuesAsOutput(_ovs), ODLA_FAILURE); //todo: duplicate set, should be failed
      CHECK_EQ(odla_GetNumOfOutputsFromComputation(comp, &_num), ODLA_SUCCESS);
      // CHECK_EQ(_num, 2); //todo: duplicate set, should be failed

      CHECK_EQ(odla_GetOutputFromComputationByIdx(comp, 0, &_ov), ODLA_SUCCESS);
      CHECK_EQ(odla_GetOutputFromComputationByIdx(comp, 2, &_ov), ODLA_INVALID_PARAM);
      // CHECK_EQ(odla_GetOutputFromComputationByIdx(nullptr, 0, &_ov), ODLA_FAILURE); //todo 1
      // CHECK_EQ(odla_GetOutputFromComputationByIdx((odla_computation)&wrong_addr, 0, &_ov), ODLA_FAILURE);

      odla_DestroyComputation(comp);
      odla_DestroyContext(ctx);

    }

    SUBCASE("test bind funtion") 
    {    
      odla_computation comp;
      odla_context ctx;
      CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
      model_helper();
      odla_CreateContext(&ctx);
      set_computationItem(comp, 1);

      int wrong_addr;
      float in = 1.f;
      float out = 1.f;

      // CHECK_EQ(odla_BindToArgumentById((const odla_value_id) "Input", &in, nullptr), ODLA_FAILURE); //todo 1
      // CHECK_EQ(odla_BindToArgumentById((const odla_value_id) "Input", &in, (odla_context)&wrong_addr), ODLA_FAILURE);
      CHECK_EQ(odla_BindToArgumentById((const odla_value_id) "Input", &in, ctx), ODLA_SUCCESS);
      // CHECK_EQ(odla_BindToArgumentById((const odla_value_id) "Input", &in, ctx), ODLA_FAILURE); // todo duplicate bind, should be recognized failure
      // CHECK_EQ(odla_BindToArgumentById((const odla_value_id) "Input", nullptr, ctx), ODLA_FAILURE); //todo 1

      // CHECK_EQ(odla_BindToOutputById((const odla_value_id) "Add", &out, nullptr), ODLA_FAILURE);
      // CHECK_EQ(odla_BindToOutputById((const odla_value_id) "Add", &out, (odla_context)&wrong_addr), ODLA_FAILURE); //todo 1
      CHECK_EQ(odla_BindToOutputById((const odla_value_id) "Add", &out, ctx), ODLA_SUCCESS);
      // CHECK_EQ(odla_BindToOutputById((const odla_value_id) "Add", &out, ctx), ODLA_FAILURE); //todo duplicate bind, should be recognized failure
      // CHECK_EQ(odla_BindToOutputById((const odla_value_id) "Add", nullptr, ctx), ODLA_FAILURE);

      odla_DestroyComputation(comp);
      odla_DestroyContext(ctx);
    }
    
    SUBCASE("test bind funtion multithread") 
    {
      odla_computation comp;
      CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
      model_helper();
      set_computationItem(comp, 1);

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

    SUBCASE("test get type function") 
    {
      odla_computation comp;
      odla_context ctx;
      CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
      model_helper();
      odla_CreateContext(&ctx);
      set_computationItem(comp, 1);

      auto _ov1 = odla_CreateArgument({ODLA_FLOAT32, {.size = 1, .dims = {1}}},
                                      (const odla_value_id)("_ov1"));
      odla_value _ov2;
      odla_value_type _ovt;
      CHECK_EQ(odla_GetValueType(_ov1, &_ovt), ODLA_SUCCESS);
      CHECK_EQ(_ovt.element_type, ODLA_FLOAT32);
      CHECK_EQ(_ovt.shape.size, 1);

      // CHECK_EQ(odla_GetValueType(_ov2, &_ovt), ODLA_FAILURE); // todo unvaild value 

      odla_DestroyComputation(comp);
      odla_DestroyContext(ctx);

    }

    SUBCASE("test execute function multithread") 
    {
      float in[3] = {1.f, 1.f, 1.f};
      float out[3] = {0.f};

      odla_computation comp;
      CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
      model_helper();
      set_computationItem(comp, 1);


      std::thread t1(execute_multithread, comp, &in[0], &out[0]);
      std::thread t2(execute_multithread, comp, &in[1], &out[1]);
      std::thread t3(execute_multithread, comp, &in[2], &out[2]);

      t1.join();
      t2.join();
      t3.join();

      CHECK_EQ(out[0], 5);
      CHECK_EQ(out[1], 5);
      CHECK_EQ(out[2], 5);

      odla_DestroyComputation(comp);
    }

    SUBCASE("test pipeline execute") 
    { 
      odla_computation comp;
      odla_context ctx;
      CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
      model_helper();
      odla_CreateContext(&ctx);
      set_computationItem(comp, 2);

        // Check parse_from_json
      float _amp = 0.6;
      int _batches_per_step = 256;
      int _ipu_num = 2;
      bool _save_model = false;
      const std::string _sdk_version = popart::core::packageHash();
      const std::string _version = std::string("1.0.0");
      const std::string _save_model_path =
          std::string("odla_popart_saved.onnx");
      bool _load_onnx = false;
      const std::string _load_onnx_path = std::string("not_set.onnx");
      const std::string _execution_mode = std::string("pipeline");
      const std::string _queue_type = std::string("LockFreeQueue");
      int _queue_capacity = 1024 * 1024;
      bool _debug = false;
      ExecutionMode _Queue_type = PIPELINE;

      json _config_json = default_json(
          _amp, _sdk_version, _version, _batches_per_step, _ipu_num,
          _save_model, _save_model_path, _load_onnx, _load_onnx_path,
          _execution_mode, _queue_type, _queue_capacity, _debug);

      PopartConfig::instance()->parse_from_json(_config_json);

      float in = 1.f;
      float out = 0.f;
      // odla_BindToArgumentById((const odla_value_id) "Input", &in, ctx);
      // odla_BindToOutputById((const odla_value_id) "Add", &out, ctx);

      // CHECK_EQ(odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr), ODLA_SUCCESS);
      // odla_DestroyComputation(comp);
      // odla_DestroyContext(ctx);
    }

}




