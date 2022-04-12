//===- Halo Compiler Generated File --------------------------------===//
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <ODLA/odla.h>
#include <stdio.h>

#include <array>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
//#include "ODLA/odla_common.h"
//#include "common.h"
#include "odla_popart.h"
#include "doctest.h"
#include "popart_config.h"
#include "ODLA/odla_common.h"

//#include "json.hpp"


typedef unsigned short uint16_t;

std::vector<float> py = {2};
std::vector<float> pz = {3};


using namespace std;

static odla_computation Comp;
static odla_context Ctx;
static odla_executable Exec;


odla_status model_helper() {

    bool use_ipu_model = 0;
    int ipu_num = 2;
    int batches_per_step = 2;
    odla_SetComputationItem(Comp, ODLA_USE_SIM_MODE,
                            (odla_item_value)&use_ipu_model);
    odla_SetComputationItem(Comp, ODLA_PROCESSOR_NUM, (odla_item_value)&ipu_num);
    odla_SetComputationItem(Comp, ODLA_BATCHES_PER_STEP,
                            (odla_item_value)&batches_per_step);
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
    //return comp;
}



TEST_CASE("testing base interface") {

    SUBCASE("test base function") 
    {

        int wrong_addr;
        // init computation && destory
        CHECK_EQ(odla_CreateComputation(&Comp), ODLA_SUCCESS);
        CHECK_EQ(odla_DestroyComputation(nullptr), ODLA_FAILURE);
        CHECK_EQ(odla_DestroyComputation((odla_computation)&wrong_addr), ODLA_FAILURE);
        CHECK_EQ(odla_DestroyComputation(Comp), ODLA_SUCCESS);

        // init context
        CHECK_EQ(odla_CreateContext(&Ctx), ODLA_SUCCESS);
        CHECK_EQ(odla_DestroyContext(Ctx), ODLA_SUCCESS);
        CHECK_EQ(odla_DestroyContext(nullptr), ODLA_FAILURE);
        CHECK_EQ(odla_DestroyContext((odla_context)&wrong_addr), ODLA_FAILURE);

        ////configure
        odla_item_value _test;
        odla_CreateComputation(&Comp);
        odla_CreateContext(&Ctx);
      
        CHECK_EQ(odla_SetComputationItem(nullptr, ODLA_USE_SIM_MODE, (odla_item_value)&_test), ODLA_FAILURE);
        CHECK_EQ(odla_SetComputationItem((odla_computation)&wrong_addr, ODLA_USE_SIM_MODE, (odla_item_value)&_test), ODLA_FAILURE);
        CHECK_EQ(odla_SetComputationItem(Comp, ODLA_USE_SIM_MODE, (odla_item_value)&_test), ODLA_SUCCESS);
        CHECK_EQ(odla_SetComputationItem(Comp, ODLA_LOAD_ENGINE_MODE, (odla_item_value)&_test), ODLA_UNSUPPORTED_DATATYPE);

        CHECK_EQ(odla_SetContextItem(nullptr, ODLA_ASYNC_CALLBACK_FUNC, (odla_item_value)&_test), ODLA_INVALID_PARAM);
        CHECK_EQ(odla_SetContextItem((odla_context)&wrong_addr, ODLA_ASYNC_CALLBACK_FUNC, (odla_item_value)&_test), ODLA_INVALID_PARAM);
        CHECK_EQ(odla_SetContextItem(Ctx, ODLA_ASYNC_CALLBACK_FUNC, (odla_item_value)&_test), ODLA_SUCCESS);

        odla_DestroyComputation(Comp);
        odla_DestroyContext(Ctx);
    }
      
    }



