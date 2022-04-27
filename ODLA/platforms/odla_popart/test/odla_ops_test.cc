//===- Halo Compiler Generated File --------------------------------===//
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

//#include "json.hpp"

typedef unsigned short uint16_t;
using namespace std;

static odla_computation Comp;
static odla_context Ctx;
static odla_executable Exec;

void set_computationItem() {
  bool use_ipu_model = 1;
  int ipu_num = 1;
  int batches_per_step = 1;
  odla_SetComputationItem(Comp, ODLA_USE_SIM_MODE,
                          (odla_item_value)&use_ipu_model);
  odla_SetComputationItem(Comp, ODLA_PROCESSOR_NUM, (odla_item_value)&ipu_num);
  odla_SetComputationItem(Comp, ODLA_BATCHES_PER_STEP,
                          (odla_item_value)&batches_per_step);
}

TEST_CASE("GEMM OPS TESTING") {

  SUBCASE("GEMM OPS TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem();

    auto a = odla_CreateArgument({ODLA_FLOAT32, {.size = 2, .dims = {4, 16}}},
                                 (const odla_value_id)("a"));
    auto b = odla_CreateArgument({ODLA_FLOAT32, {.size = 2, .dims = {16, 8}}},
                                 (const odla_value_id)("b"));

    auto result =
        odla_Gemm(a, 0, b, 0, 1, 0, nullptr, {.size = 2, .dims = {4, 8}},
                  (const odla_value_id) "Gemm");
    odla_SetValueAsOutput(result);

    static odla_context ctx;
    odla_CreateContext(&ctx);

    std::vector<float> a_data(4 * 16, 1.0);
    odla_BindToArgumentById((const odla_value_id) "a", a_data.data(), ctx);

    std::vector<float> b_data(16 * 8, 1.0);
    odla_BindToArgumentById((const odla_value_id) "b", b_data.data(), ctx);

    std::vector<float> result_data(4 * 8, 0);
    std::vector<float> expected(4 * 8, 16.0);
    odla_BindToOutputById((const odla_value_id) "Gemm", result_data.data(),
                          ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

    // BOOST_TEST_MESSAGE("result_data = " << test::VecToStr(result_data));

    for (int i = 0; i < expected.size(); i++) {
      CHECK_EQ(result_data[i], expected[i]);
    }

    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
  }

  SUBCASE("GEMM TRANSPOSE TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem();

    auto a = odla_CreateArgument(
        {ODLA_FLOAT32, {.size = 4, .dims = {1, 12, 64, 64}}},
        (const odla_value_id)("a"));
    auto b = odla_CreateArgument(
        {ODLA_FLOAT32, {.size = 4, .dims = {1, 12, 64, 64}}},
        (const odla_value_id)("b"));

    auto result = odla_Gemm(a, 0, b, 1, 1, 0, nullptr, {.size = 0, .dims = {0}},
                            (const odla_value_id) "Gemm");

    odla_SetValueAsOutput(result);

    static odla_context ctx;
    odla_CreateContext(&ctx);

    std::vector<float> a_data(1 * 12 * 64 * 64, 1.0);
    odla_BindToArgumentById((const odla_value_id) "a", a_data.data(), ctx);

    std::vector<float> b_data(1 * 12 * 64 * 64, 1.0);
    odla_BindToArgumentById((const odla_value_id) "b", b_data.data(), ctx);

    std::vector<float> result_data(1 * 12 * 64 * 64, 0);

    odla_BindToOutputById((const odla_value_id) "Gemm", result_data.data(),
                          ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
  }

}

