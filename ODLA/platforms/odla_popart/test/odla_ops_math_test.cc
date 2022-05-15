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

TEST_CASE("MATH OPS TESTING") {
    
    SUBCASE("MATH OPS ABS TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem();


    auto input =
        odla_CreateArgument({ODLA_FLOAT32, {.size = 2, .dims = {2, 2}}},
                            (const odla_value_id)("input"));

    auto abs_value = odla_Abs(input, (const odla_value_id) "Abs");
    odla_SetValueAsOutput(abs_value);

    static odla_context ctx;
    odla_CreateContext(&ctx);

    float input_data[2 * 2] = {2.0, 1.0, -3.0, -10.0};
    odla_BindToArgumentById((const odla_value_id) "input", input_data, ctx);

    float out_Abs[2 * 2] = {0, 0, 0, 0};
    odla_BindToOutputById((const odla_value_id) "Abs", out_Abs, ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

    std::cout << "out_Abs = [";
    for (int i = 0; i < 4; i++) {
      std::cout << out_Abs[i] << ", ";
    }
    std::cout << "]" << std::endl;

    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);

  }
    
    SUBCASE("MATH OPS ARG MIN TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem();

    auto input =
        odla_CreateArgument({ODLA_FLOAT32, {.size = 2, .dims = {2, 2}}},
                            (const odla_value_id)("input"));
    odla_int32 axis = 0;
    odla_bool keep_dims = 1;
    odla_bool return_last_index = 0;

    odla_value_type unused_arg;
    auto ArgMin = odla_ArgMin(input, axis, keep_dims, return_last_index,
                              unused_arg, (const odla_value_id) "ArgMin");
    odla_SetValueAsOutput(ArgMin);

    static odla_context ctx;
    odla_CreateContext(&ctx);

    float input_data[2 * 2] = {2.0, 1.0, 3.0, 10.0};
    odla_BindToArgumentById((const odla_value_id) "input", input_data, ctx);

    int32_t out_ArgMin[2] = {-1, -1};
    odla_BindToOutputById((const odla_value_id) "ArgMin", out_ArgMin, ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

    std::cout << "out_ArgMin = [";
    for (int i = 0; i < 2; i++) {
      std::cout << out_ArgMin[i] << ", ";
    }
    std::cout << "]" << std::endl;
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
  }
    
    SUBCASE("MATH OPS CEIL TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem();


    auto input =
        odla_CreateArgument({ODLA_FLOAT32, {.size = 2, .dims = {2, 2}}},
                            (const odla_value_id)("input"));

    auto Ceil = odla_Ceil(input, (const odla_value_id) "Ceil");
    odla_SetValueAsOutput(Ceil);

    static odla_context ctx;
    odla_CreateContext(&ctx);

    float input_data[2 * 2] = {2.0, 1.0, 3.5, 10.0};
    odla_BindToArgumentById((const odla_value_id) "input", input_data, ctx);

    float out_Ceil[4] = {0, 0, 0, 0};
    odla_BindToOutputById((const odla_value_id) "Ceil", out_Ceil, ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

    std::cout << "out_Ceil = [";
    for (int i = 0; i < 4; i++) {
      std::cout << out_Ceil[i] << ", ";
    }
    std::cout << "]" << std::endl;
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
  }

    SUBCASE("MATH OPS CLAMP TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem();


    auto input =
        odla_CreateArgument({ODLA_FLOAT32, {.size = 2, .dims = {3, 3}}},
                            (const odla_value_id)("input"));

    float lo_data = 3.0;
    float hi_data = 5.0;

    auto Clamp =
        odla_Clamp(input, lo_data, hi_data, (const odla_value_id) "Clamp");
    odla_SetValueAsOutput(Clamp);

    static odla_context ctx;
    odla_CreateContext(&ctx);

    float input_data[3 * 3] = {2.0, 1.0, 3.5, 10.0, 4.3, 5.8, 9.0, 12.0, 100.3};
    odla_BindToArgumentById((const odla_value_id) "input", input_data, ctx);

    float out_Clamp[3 * 3] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    odla_BindToOutputById((const odla_value_id) "Clamp", out_Clamp, ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

    std::cout << "out_Clamp = [";
    for (int i = 0; i < 9; i++) {
        std::cout << out_Clamp[i] << ", ";
    }
    std::cout << "]" << std::endl;
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
    }

    SUBCASE("MATH OPS EQUAL TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem();

    auto lhs = odla_CreateArgument({ODLA_FLOAT32, {.size = 2, .dims = {2, 2}}},
                                    (const odla_value_id)("lhs"));

    auto rhs = odla_CreateArgument({ODLA_FLOAT32, {.size = 2, .dims = {2, 2}}},
                                    (const odla_value_id)("rhs"));

    auto Equal = odla_Equal(lhs, rhs, (const odla_value_id) "Equal");
    odla_SetValueAsOutput(Equal);

    static odla_context ctx;
    odla_CreateContext(&ctx);

    float lhs_data[2 * 2] = {2.0, 1.0, 3.5, 10.0};
    odla_BindToArgumentById((const odla_value_id) "lhs", lhs_data, ctx);

    float rhs_data[2 * 2] = {2.0, 1.0, 3.5, 9.0};
    odla_BindToArgumentById((const odla_value_id) "rhs", rhs_data, ctx);

    bool out_Equal[2 * 2] = {false, false, false, false};
    odla_BindToOutputById((const odla_value_id) "Equal", out_Equal, ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

    std::cout << "out_Equal = [";
    for (int i = 0; i < 4; i++) {
        std::cout << out_Equal[i] << ", ";
    }
    std::cout << "]" << std::endl;
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
    }
    
    SUBCASE("MATH OPS EXP TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem();

    auto input =
        odla_CreateArgument({ODLA_FLOAT32, {.size = 2, .dims = {2, 2}}},
                            (const odla_value_id)("input"));

    auto Exp = odla_Exp(input, (const odla_value_id) "Exp");
    odla_SetValueAsOutput(Exp);

    static odla_context ctx;
    odla_CreateContext(&ctx);

    float input_data[2 * 2] = {2.0, 1.0, 3.5, 5.0};
    odla_BindToArgumentById((const odla_value_id) "input", input_data, ctx);

    float out_Exp[4] = {0, 0, 0, 0};
    odla_BindToOutputById((const odla_value_id) "Exp", out_Exp, ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

    std::cout << "out_Exp = [";
    for (int i = 0; i < 4; i++) {
        std::cout << out_Exp[i] << ", ";
    }
    std::cout << "]" << std::endl;
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
    }
   
    SUBCASE("MATH OPS GREATER TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem();

    auto lhs = odla_CreateArgument({ODLA_FLOAT32, {.size = 2, .dims = {2, 2}}},
                                    (const odla_value_id)("lhs"));

    auto rhs = odla_CreateArgument({ODLA_FLOAT32, {.size = 2, .dims = {2, 2}}},
                                    (const odla_value_id)("rhs"));

    auto Greater = odla_Greater(lhs, rhs, (const odla_value_id) "Greater");
    odla_SetValueAsOutput(Greater);

    static odla_context ctx;
    odla_CreateContext(&ctx);

    float lhs_data[2 * 2] = {2.3, 1.0, 3.5, 5.5};
    odla_BindToArgumentById((const odla_value_id) "lhs", lhs_data, ctx);

    float rhs_data[2 * 2] = {2.0, 1.5, 4.5, 5.0};
    odla_BindToArgumentById((const odla_value_id) "rhs", rhs_data, ctx);

    bool out_Greater[4] = {0, 0, 0, 0};
    odla_BindToOutputById((const odla_value_id) "Greater", out_Greater, ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

    std::cout << "out_Greater = [";
    for (int i = 0; i < 4; i++) {
        std::cout << out_Greater[i] << ", ";
    }
    std::cout << "]" << std::endl;
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
    }
   
    SUBCASE("MATH OPS LESS TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem();

    auto lhs = odla_CreateArgument({ODLA_FLOAT32, {.size = 2, .dims = {2, 2}}},
                                    (const odla_value_id)("lhs"));

    auto rhs = odla_CreateArgument({ODLA_FLOAT32, {.size = 2, .dims = {2, 2}}},
                                    (const odla_value_id)("rhs"));

    auto Less = odla_Less(lhs, rhs, (const odla_value_id) "Less");
    odla_SetValueAsOutput(Less);

    static odla_context ctx;
    odla_CreateContext(&ctx);

    float lhs_data[2 * 2] = {2.3, 1.0, 3.5, 5.5};
    odla_BindToArgumentById((const odla_value_id) "lhs", lhs_data, ctx);

    float rhs_data[2 * 2] = {2.0, 1.5, 4.5, 5.0};
    odla_BindToArgumentById((const odla_value_id) "rhs", rhs_data, ctx);

    bool out_Less[4] = {0, 0, 0, 0};
    odla_BindToOutputById((const odla_value_id) "Less", out_Less, ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

    std::cout << "out_Less = [";
    for (int i = 0; i < 4; i++) {
        std::cout << out_Less[i] << ", ";
    }
    std::cout << "]" << std::endl;
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
    }
   
    SUBCASE("MATH OPS LOG TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem();

    auto input =
        odla_CreateArgument({ODLA_FLOAT32, {.size = 2, .dims = {2, 2}}},
                            (const odla_value_id)("input"));

    auto Log = odla_Log(input, (const odla_value_id) "Log");
    odla_SetValueAsOutput(Log);

    static odla_context ctx;
    odla_CreateContext(&ctx);

    float input_data[2 * 2] = {2.0, 1.0, 3.5, 5.0};
    odla_BindToArgumentById((const odla_value_id) "input", input_data, ctx);

    float out_Log[4] = {0, 0, 0, 0};
    odla_BindToOutputById((const odla_value_id) "Log", out_Log, ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

    std::cout << "out_Log = [";
    for (int i = 0; i < 4; i++) {
        std::cout << out_Log[i] << ", ";
    }
    std::cout << "]" << std::endl;
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
    }

    SUBCASE("MATH OPS MAX TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem();

    auto lhs = odla_CreateArgument({ODLA_FLOAT32, {.size = 2, .dims = {2, 2}}},
                                    (const odla_value_id)("lhs"));

    auto rhs = odla_CreateArgument({ODLA_FLOAT32, {.size = 2, .dims = {2, 2}}},
                                    (const odla_value_id)("rhs"));

    auto Max = odla_Max(lhs, rhs, (const odla_value_id) "Max");
    odla_SetValueAsOutput(Max);

    static odla_context ctx;
    odla_CreateContext(&ctx);

    float lhs_data[2 * 2] = {2.0, 1.0, 3.5, 5.0};
    odla_BindToArgumentById((const odla_value_id) "lhs", lhs_data, ctx);

    float rhs_data[2 * 2] = {4.0, 0.9, -3.5, 588888.0};
    odla_BindToArgumentById((const odla_value_id) "rhs", rhs_data, ctx);

    float out_Max[4] = {0, 0, 0, 0};
    odla_BindToOutputById((const odla_value_id) "Max", out_Max, ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

    std::cout << "out_Max = [";
    for (int i = 0; i < 4; i++) {
        std::cout << out_Max[i] << ", ";
    }
    std::cout << "]" << std::endl;
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
    }

    SUBCASE("MATH OPS MIN TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem();

    auto lhs = odla_CreateArgument({ODLA_FLOAT32, {.size = 2, .dims = {2, 2}}},
                                    (const odla_value_id)("lhs"));

    auto rhs = odla_CreateArgument({ODLA_FLOAT32, {.size = 2, .dims = {2, 2}}},
                                    (const odla_value_id)("rhs"));

    auto Min = odla_Min(lhs, rhs, (const odla_value_id) "Min");
    odla_SetValueAsOutput(Min);

    static odla_context ctx;
    odla_CreateContext(&ctx);

    float lhs_data[2 * 2] = {2.0, 1.0, 3.5, 5.0};
    odla_BindToArgumentById((const odla_value_id) "lhs", lhs_data, ctx);

    float rhs_data[2 * 2] = {4.0, 0.9, -3.5, 588888.0};
    odla_BindToArgumentById((const odla_value_id) "rhs", rhs_data, ctx);

    float out_Min[4] = {0, 0, 0, 0};
    odla_BindToOutputById((const odla_value_id) "Min", out_Min, ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

    std::cout << "out_Min = [";
    for (int i = 0; i < 4; i++) {
        std::cout << out_Min[i] << ", ";
    }
    std::cout << "]" << std::endl;
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
    }

    SUBCASE("MATH OPS MEAN TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem();

    auto input_1 =
        odla_CreateArgument({ODLA_FLOAT32, {.size = 2, .dims = {2, 2}}},
                            (const odla_value_id)("input_1"));

    auto input_2 =
        odla_CreateArgument({ODLA_FLOAT32, {.size = 2, .dims = {2, 2}}},
                            (const odla_value_id)("input_2"));

    auto input_3 =
        odla_CreateArgument({ODLA_FLOAT32, {.size = 2, .dims = {2, 2}}},
                            (const odla_value_id)("input_3"));

    //  std::vector<odla_value> input_vec{input_1, input_2, input_3};
    odla_values inputs{.size = 3, .values = {input_1, input_2, input_3}};
    auto Mean = odla_Mean(inputs, (const odla_value_id) "Mean");
    odla_SetValueAsOutput(Mean);

    static odla_context ctx;
    odla_CreateContext(&ctx);

    std::vector<float> input_data_1 = {0.1, 0.2, 0.3, 0.4};
    odla_BindToArgumentById((const odla_value_id) "input_1",
                            input_data_1.data(), ctx);

    std::vector<float> input_data_2 = {0.5, 0.6, 0.7, 0.8};
    odla_BindToArgumentById((const odla_value_id) "input_2",
                            input_data_2.data(), ctx);

    std::vector<float> input_data_3 = {0.9, 1.0, 1.5, 1.8};
    odla_BindToArgumentById((const odla_value_id) "input_3",
                            input_data_3.data(), ctx);
    // float input_data[2 * 2] = {2.0, 1.0, 3.5, 5.0};
    // odla_BindToArgumentById((const odla_value_id) "input", input_data, ctx);

    float out_Mean[4] = {0, 0, 0, 0};
    odla_BindToOutputById((const odla_value_id) "Mean", out_Mean, ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

    std::cout << "out_Mean = [";
    for (int i = 0; i < 4; i++) {
        std::cout << out_Mean[i] << ", ";
    }
    std::cout << "]" << std::endl;
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
    }

    SUBCASE("MATH OPS NEG TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem();

    auto input =
        odla_CreateArgument({ODLA_FLOAT32, {.size = 2, .dims = {2, 2}}},
                            (const odla_value_id)("input"));

    auto Neg = odla_Neg(input, (const odla_value_id) "Neg");
    odla_SetValueAsOutput(Neg);

    static odla_context ctx;
    odla_CreateContext(&ctx);

    float input_data[2 * 2] = {2.0, 1.0, 3.5, 5.0};
    odla_BindToArgumentById((const odla_value_id) "input", input_data, ctx);

    float out_Neg[4] = {0, 0, 0, 0};
    odla_BindToOutputById((const odla_value_id) "Neg", out_Neg, ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

    std::cout << "out_Neg = [";
    for (int i = 0; i < 4; i++) {
        std::cout << out_Neg[i] << ", ";
    }
    std::cout << "]" << std::endl;
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
    }

    SUBCASE("MATH OPS NOT TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem();

    auto input = odla_CreateArgument({ODLA_BOOL, {.size = 2, .dims = {2, 2}}},
                                        (const odla_value_id)("input"));

    auto Not = odla_Not(input, (const odla_value_id) "Not");
    odla_SetValueAsOutput(Not);

    static odla_context ctx;
    odla_CreateContext(&ctx);

    bool input_data[2 * 2] = {true, false, true, false};
    odla_BindToArgumentById((const odla_value_id) "input", input_data, ctx);

    bool out_Not[4] = {false, false, false, false};
    odla_BindToOutputById((const odla_value_id) "Not", out_Not, ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

    std::cout << "out_Not = [";
    for (int i = 0; i < 4; i++) {
        std::cout << out_Not[i] << ", ";
    }
    std::cout << "]" << std::endl;
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
    }

    SUBCASE("MATH OPS POW TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem();

    auto lhs = odla_CreateArgument({ODLA_FLOAT32, {.size = 2, .dims = {2, 2}}},
                                    (const odla_value_id)("lhs"));

    auto rhs = odla_CreateArgument({ODLA_FLOAT32, {.size = 2, .dims = {2, 2}}},
                                    (const odla_value_id)("rhs"));

    auto Pow = odla_Pow(lhs, rhs, (const odla_value_id) "Pow");
    odla_SetValueAsOutput(Pow);

    static odla_context ctx;
    odla_CreateContext(&ctx);

    float lhs_data[2 * 2] = {2.0, 1.0, 3.5, 5.0};
    odla_BindToArgumentById((const odla_value_id) "lhs", lhs_data, ctx);

    float rhs_data[2 * 2] = {4.0, 0.9, -2.0, 4.0};
    odla_BindToArgumentById((const odla_value_id) "rhs", rhs_data, ctx);

    float out_Pow[4] = {0, 0, 0, 0};
    odla_BindToOutputById((const odla_value_id) "Pow", out_Pow, ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

    std::cout << "out_Pow = [";
    for (int i = 0; i < 4; i++) {
        std::cout << out_Pow[i] << ", ";
    }
    std::cout << "]" << std::endl;
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
    }
   
    SUBCASE("MATH OPS RECIPROCAL TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem();

    auto input =
        odla_CreateArgument({ODLA_FLOAT32, {.size = 2, .dims = {2, 2}}},
                            (const odla_value_id)("input"));

    auto Reciprocal =
        odla_Reciprocal(input, (const odla_value_id) "Reciprocal");
    odla_SetValueAsOutput(Reciprocal);

    static odla_context ctx;
    odla_CreateContext(&ctx);

    float input_data[2 * 2] = {-0.2, -4, 8, 9};
    odla_BindToArgumentById((const odla_value_id) "input", input_data, ctx);

    float out_Reciprocal[4] = {0, 0, 0, 0};
    odla_BindToOutputById((const odla_value_id) "Reciprocal", out_Reciprocal,
                            ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

    std::cout << "out_Reciprocal = [";
    for (int i = 0; i < 4; i++) {
        std::cout << out_Reciprocal[i] << ", ";
    }
    std::cout << "]" << std::endl;
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
    }
   
    SUBCASE("MATH OPS REDUCEMAX TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem();

    auto input =
        odla_CreateArgument({ODLA_FLOAT32, {.size = 2, .dims = {2, 2}}},
                            (const odla_value_id)("input"));

    odla_size_t num_of_axes = 1;
    odla_bool keep_dims = 1;
    odla_uint32 axes[1] = {1};
    odla_value_shape output_dims;

    auto ReduceMax =
        odla_ReduceMax(input, num_of_axes, axes, keep_dims, output_dims,
                        (const odla_value_id) "ReduceMax");
    odla_SetValueAsOutput(ReduceMax);

    static odla_context ctx;
    odla_CreateContext(&ctx);

    float input_data[2 * 2] = {-0.2, -4, 8, 9};
    odla_BindToArgumentById((const odla_value_id) "input", input_data, ctx);

    float out_ReduceMax[4] = {0, 0, 0, 0};
    odla_BindToOutputById((const odla_value_id) "ReduceMax", out_ReduceMax,
                            ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

    std::cout << "out_ReduceMax = [";
    for (int i = 0; i < 4; i++) {
        std::cout << out_ReduceMax[i] << ", ";
    }
    std::cout << "]" << std::endl;
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
    }
   
    SUBCASE("MATH OPS REDUCEMIN TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem();

    auto input =
        odla_CreateArgument({ODLA_FLOAT32, {.size = 2, .dims = {2, 2}}},
                            (const odla_value_id)("input"));

    odla_size_t num_of_axes = 1;
    odla_bool keep_dims = 2;
    odla_uint32 axes[2] = {1, 0};
    odla_value_shape output_dims;

    auto ReduceMin =
        odla_ReduceMin(input, num_of_axes, axes, keep_dims, output_dims,
                        (const odla_value_id) "ReduceMin");
    odla_SetValueAsOutput(ReduceMin);

    static odla_context ctx;
    odla_CreateContext(&ctx);

    float input_data[2 * 2] = {-0.2, -4, 8, 9};
    odla_BindToArgumentById((const odla_value_id) "input", input_data, ctx);

    float out_ReduceMin[4] = {0, 0, 0, 0};
    odla_BindToOutputById((const odla_value_id) "ReduceMin", out_ReduceMin,
                            ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

    std::cout << "out_ReduceMin = [";
    for (int i = 0; i < 4; i++) {
        std::cout << out_ReduceMin[i] << ", ";
    }
    std::cout << "]" << std::endl;
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
    }
   
    SUBCASE("MATH OPS REDUCEPROD TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem();

    auto input =
        odla_CreateArgument({ODLA_FLOAT32, {.size = 2, .dims = {2, 2}}},
                            (const odla_value_id)("input"));

    odla_size_t num_of_axes = 1;
    odla_bool keep_dims = 2;
    odla_uint32 axes[1] = {0};
    odla_value_shape output_dims;

    auto ReduceProd =
        odla_ReduceProd(input, num_of_axes, axes, keep_dims, output_dims,
                        (const odla_value_id) "ReduceProd");
    odla_SetValueAsOutput(ReduceProd);

    static odla_context ctx;
    odla_CreateContext(&ctx);

    float input_data[2 * 2] = {-0.2, -4, 8, 9};
    odla_BindToArgumentById((const odla_value_id) "input", input_data, ctx);

    float out_ReduceProd[4] = {0, 0, 0, 0};
    odla_BindToOutputById((const odla_value_id) "ReduceProd", out_ReduceProd,
                            ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

    std::cout << "out_ReduceProd = [";
    for (int i = 0; i < 4; i++) {
        std::cout << out_ReduceProd[i] << ", ";
    }
    std::cout << "]" << std::endl;
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
    }
   
    SUBCASE("MATH OPS REDUCESUM TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem();

    auto input =
        odla_CreateArgument({ODLA_FLOAT32, {.size = 2, .dims = {2, 2}}},
                            (const odla_value_id)("input"));

    odla_size_t num_of_axes = 1;
    odla_bool keep_dims = 2;
    odla_uint32 axes[1] = {0};
    odla_value_shape output_dims;

    auto ReduceSum =
        odla_ReduceSum(input, num_of_axes, axes, keep_dims, output_dims,
                        (const odla_value_id) "ReduceSum");
    odla_SetValueAsOutput(ReduceSum);

    static odla_context ctx;
    odla_CreateContext(&ctx);

    float input_data[2 * 2] = {-0.2, -4, 8, 9};
    odla_BindToArgumentById((const odla_value_id) "input", input_data, ctx);

    float out_ReduceSum[4] = {0, 0, 0, 0};
    odla_BindToOutputById((const odla_value_id) "ReduceSum", out_ReduceSum,
                            ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

    std::cout << "out_ReduceSum = [";
    for (int i = 0; i < 4; i++) {
        std::cout << out_ReduceSum[i] << ", ";
    }
    std::cout << "]" << std::endl;
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
    }

    SUBCASE("MATH OPS SIGN TEST") {
      odla_computation comp;
      CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
      set_computationItem();

      auto input =
          odla_CreateArgument({ODLA_FLOAT32, {.size = 2, .dims = {2, 2}}},
                              (const odla_value_id)("input"));

      auto Sign = odla_Sign(input, (const odla_value_id) "Sign");
      odla_SetValueAsOutput(Sign);

      static odla_context ctx;
      odla_CreateContext(&ctx);

      float input_data[2 * 2] = {-0.2, -0.3, 1, 0.5};
      odla_BindToArgumentById((const odla_value_id) "input", input_data, ctx);

      float out_Sign[4] = {0, 0, 0, 0};
      odla_BindToOutputById((const odla_value_id) "Sign", out_Sign, ctx);

      odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

      std::cout << "out_Sign = [";
      for (int i = 0; i < 4; i++) {
        std::cout << out_Sign[i] << ", ";
      }
      std::cout << "]" << std::endl;

      odla_DestroyComputation(comp);
      odla_DestroyContext(ctx);

    }

    SUBCASE("MATH OPS AND TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem(comp, 1);

    auto lhs = odla_CreateArgument({ODLA_BOOL, {.size = 2, .dims = {2, 2}}},
                                    (const odla_value_id)("lhs"));

    auto rhs = odla_CreateArgument({ODLA_BOOL, {.size = 2, .dims = {2, 2}}},
                                    (const odla_value_id)("rhs"));

    auto And = odla_And(lhs, rhs, (const odla_value_id) "And");
    odla_SetValueAsOutput(And);

    static odla_context ctx;
    odla_CreateContext(&ctx);

    bool lhs_data[2 * 2] = {false, false, true, true};
    odla_BindToArgumentById((const odla_value_id) "lhs", lhs_data, ctx);

    bool rhs_data[2 * 2] = {true, false, true, false};
    odla_BindToArgumentById((const odla_value_id) "rhs", rhs_data, ctx);

    bool out_And[2 * 2] = {false, false, false, false};
    odla_BindToOutputById((const odla_value_id) "And", out_And, ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

    std::cout << "out_And = [";
    for (int i = 0; i < 4; i++) {
        std::cout << out_And[i] << ", ";
    }
    std::cout << "]" << std::endl;
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
    }
  
    SUBCASE("MATH OPS OR TEST"){
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem(comp, 1);

    auto lhs = odla_CreateArgument({ODLA_BOOL, {.size = 2, .dims = {2, 2}}},
                                    (const odla_value_id)("lhs"));

    auto rhs = odla_CreateArgument({ODLA_BOOL, {.size = 2, .dims = {2, 2}}},
                                    (const odla_value_id)("rhs"));

    auto Or = odla_Or(lhs, rhs, (const odla_value_id) "Or");
    odla_SetValueAsOutput(Or);

    static odla_context ctx;
    odla_CreateContext(&ctx);

    bool lhs_data[2 * 2] = {false, false, true, true};
    odla_BindToArgumentById((const odla_value_id) "lhs", lhs_data, ctx);

    bool rhs_data[2 * 2] = {true, false, true, false};
    odla_BindToArgumentById((const odla_value_id) "rhs", rhs_data, ctx);

    bool out_Or[2 * 2] = {false, false, false, false};
    odla_BindToOutputById((const odla_value_id) "Or", out_Or, ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

    std::cout << "out_Or = [";
    for (int i = 0; i < 4; i++) {
        std::cout << out_Or[i] << ", ";
    }
    std::cout << "]" << std::endl;
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
    }
    
    SUBCASE("MATH OPS NOT EQUAL TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem(comp, 1);

    auto lhs = odla_CreateArgument({ODLA_BOOL, {.size = 2, .dims = {2, 2}}},
                                    (const odla_value_id)("lhs"));

    auto rhs = odla_CreateArgument({ODLA_BOOL, {.size = 2, .dims = {2, 2}}},
                                    (const odla_value_id)("rhs"));

    auto NotEqual = odla_NotEqual(lhs, rhs, (const odla_value_id) "NotEqual");
    odla_SetValueAsOutput(NotEqual);

    static odla_context ctx;
    odla_CreateContext(&ctx);

    bool lhs_data[2 * 2] = {false, false, true, true};
    odla_BindToArgumentById((const odla_value_id) "lhs", lhs_data, ctx);

    bool rhs_data[2 * 2] = {true, false, true, false};
    odla_BindToArgumentById((const odla_value_id) "rhs", rhs_data, ctx);

    bool out_Equal[2 * 2] = {false, false, false, false};
    odla_BindToOutputById((const odla_value_id) "NotEqual", out_Equal, ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

    std::cout << "out_Equal = [";
    for (int i = 0; i < 4; i++) {
        std::cout << out_Equal[i] << ", ";
    }
    std::cout << "]" << std::endl;
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
    }
}

