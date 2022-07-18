//===- odla_ops_math_test.cc
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
//#include "json.hpp"

typedef unsigned short uint16_t;
using namespace std;

TEST_CASE("MATH OPS TESTING") {
  SUBCASE("MATH OPS ABS TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem(comp);

    auto input =
        odla_CreateArgument({ODLA_FLOAT32, {.size = 2, .dims = {2, 2}}},
                            (const odla_value_id)("input"));

    auto abs_value = odla_Abs(input, (const odla_value_id) "Abs");
    odla_SetValueAsOutput(abs_value);

    odla_context ctx;
    odla_CreateContext(&ctx);

    std::vector<float> input_data = {2.0, 1.0, -3.0, -10.0};
    odla_BindToArgumentById((const odla_value_id) "input", input_data.data(),
                            ctx);

    std::vector<float> out_Abs(4);
    odla_BindToOutputById((const odla_value_id) "Abs", out_Abs.data(), ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

    std::vector<float> expected = {
        2,
        1,
        3,
        10,
    };
    CHECK_EQ(expected, out_Abs);

    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
  }

  SUBCASE("MATH OPS ARG MIN TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem(comp);

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
    std::vector<float> input_data = {2.0, 1.0, -3.0, -10.0};
    odla_BindToArgumentById((const odla_value_id) "input", input_data.data(),
                            ctx);

    std::vector<int32_t> out_ArgMin(2);
    odla_BindToOutputById((const odla_value_id) "ArgMin", out_ArgMin.data(),
                          ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

    std::vector<int32_t> expected = {1, 1};
    CHECK_EQ(expected, out_ArgMin);
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
  }

  SUBCASE("MATH OPS CEIL TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem(comp);

    auto input =
        odla_CreateArgument({ODLA_FLOAT32, {.size = 2, .dims = {2, 2}}},
                            (const odla_value_id)("input"));

    auto Ceil = odla_Ceil(input, (const odla_value_id) "Ceil");
    odla_SetValueAsOutput(Ceil);

    static odla_context ctx;
    odla_CreateContext(&ctx);

    float input_data[2 * 2] = {2.0, 1.0, 3.5, 10.0};
    odla_BindToArgumentById((const odla_value_id) "input", input_data, ctx);

    std::vector<float> out_Ceil(4);
    odla_BindToOutputById((const odla_value_id) "Ceil", out_Ceil.data(), ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

    std::vector<float> expected = {2, 1, 4, 10};
    CHECK_EQ(expected, out_Ceil);

    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
  }

  SUBCASE("MATH OPS CLAMP TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem(comp);

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
    std::vector<float> out_Clamp(9);
    odla_BindToOutputById((const odla_value_id) "Clamp", out_Clamp.data(), ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);
    std::vector<float> expected = {3, 3, 3.5, 5, 4.3, 5, 5, 5, 5};
    CHECK_EQ(expected, out_Clamp);
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
  }

  SUBCASE("MATH OPS EQUAL TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem(comp);

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
    for (int i = 0; i < 4; i++) {
      CHECK_EQ(out_Equal[i], out_Equal[i]);
    }
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
  }

  SUBCASE("MATH OPS EXP TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem(comp);

    auto input =
        odla_CreateArgument({ODLA_FLOAT32, {.size = 2, .dims = {2, 2}}},
                            (const odla_value_id)("input"));

    auto Exp = odla_Exp(input, (const odla_value_id) "Exp");
    odla_SetValueAsOutput(Exp);

    static odla_context ctx;
    odla_CreateContext(&ctx);

    float input_data[2 * 2] = {2.0, 1.0, 3.5, 5.0};
    odla_BindToArgumentById((const odla_value_id) "input", input_data, ctx);

    std::vector<float> out_Exp(4);
    odla_BindToOutputById((const odla_value_id) "Exp", out_Exp.data(), ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);
    std::vector<float> expected = {7.38906, 2.71828, 33.1155, 148.413};
    CHECK_EQ(expected, expected);
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
  }

  SUBCASE("MATH OPS GREATER TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem(comp);

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

    std::vector<float> expected = {1, 0, 0, 1};
    for (int i = 0; i < 4; i++) {
      CHECK_EQ(expected[i], out_Greater[i]);
    }
    std::cout << "]" << std::endl;
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
  }

  SUBCASE("MATH OPS LESS TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem(comp);

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

    std::vector<float> expected = {0, 1, 1, 0};
    for (int i = 0; i < 4; i++) {
      CHECK_EQ(expected[i], out_Less[i]);
    }
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
  }

  SUBCASE("MATH OPS LOG TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem(comp);

    auto input =
        odla_CreateArgument({ODLA_FLOAT32, {.size = 2, .dims = {2, 2}}},
                            (const odla_value_id)("input"));

    auto Log = odla_Log(input, (const odla_value_id) "Log");
    odla_SetValueAsOutput(Log);

    static odla_context ctx;
    odla_CreateContext(&ctx);

    float input_data[2 * 2] = {2.0, 1.0, 3.5, 5.0};
    odla_BindToArgumentById((const odla_value_id) "input", input_data, ctx);

    std::vector<float> out_Log(4);
    odla_BindToOutputById((const odla_value_id) "Log", out_Log.data(), ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);
    std::vector<float> expected = {0.693147, 0, 1.25276, 1.60944};
    for (int i = 0; i < 4; i++) {
      CHECK_LT(abs(expected[i] - out_Log[i]), TOLLERANCE);
    }

    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
  }

  SUBCASE("MATH OPS MAX TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem(comp);

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

    float rhs_data[2 * 2] = {4.0, 0.9, -3.5, 58.0};
    odla_BindToArgumentById((const odla_value_id) "rhs", rhs_data, ctx);

    std::vector<float> out_Max(4);
    odla_BindToOutputById((const odla_value_id) "Max", out_Max.data(), ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);
    std::vector<float> expected = {4, 1, 3.5, 58};
    CHECK_EQ(expected, out_Max);
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
  }

  SUBCASE("MATH OPS MIN TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem(comp);

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

    std::vector<float> out_Min(4);
    odla_BindToOutputById((const odla_value_id) "Min", out_Min.data(), ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);
    std::vector<float> expected = {2, 0.9, -3.5, 5};
    for (int i = 0; i < 4; i++) {
      CHECK_LT(abs(expected[i] - out_Min[i]), TOLLERANCE);
    }

    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
  }

  SUBCASE("MATH OPS MEAN TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem(comp);

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
    std::vector<float> out_Mean(4);
    odla_BindToOutputById((const odla_value_id) "Mean", out_Mean.data(), ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);
    std::vector<float> expected = {0.5, 0.6, 0.833333, 1};
    for (int i = 0; i < 4; i++) {
      CHECK_LT(abs(expected[i] - out_Mean[i]), TOLLERANCE);
    }
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
  }

  SUBCASE("MATH OPS NEG TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem(comp);

    auto input =
        odla_CreateArgument({ODLA_FLOAT32, {.size = 2, .dims = {2, 2}}},
                            (const odla_value_id)("input"));

    auto Neg = odla_Neg(input, (const odla_value_id) "Neg");
    odla_SetValueAsOutput(Neg);

    static odla_context ctx;
    odla_CreateContext(&ctx);

    float input_data[2 * 2] = {2.0, 1.0, 3.5, 5.0};
    odla_BindToArgumentById((const odla_value_id) "input", input_data, ctx);
    std::vector<float> out_Neg(4);
    odla_BindToOutputById((const odla_value_id) "Neg", out_Neg.data(), ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);
    std::vector<float> expected = {-2, -1, -3.5, -5};
    CHECK_EQ(expected, out_Neg);
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
  }

  SUBCASE("MATH OPS NOT TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem(comp);

    auto input = odla_CreateArgument({ODLA_BOOL, {.size = 2, .dims = {2, 2}}},
                                     (const odla_value_id)("input"));

    auto Not = odla_Not(input, (const odla_value_id) "Not");
    odla_SetValueAsOutput(Not);

    static odla_context ctx;
    odla_CreateContext(&ctx);

    bool input_data[2 * 2] = {true, false, true, false};
    odla_BindToArgumentById((const odla_value_id) "input", input_data, ctx);

    bool out_Not[4] = {false, false, false, false};

    bool expected[4] = {false, true, false, true};
    odla_BindToOutputById((const odla_value_id) "Not", out_Not, ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

    for (int i = 0; i < 4; i++) {
      CHECK_EQ(out_Not[i], expected[i]);
    }
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
  }

  SUBCASE("MATH OPS POW TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem(comp);

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
    std::vector<float> out_Pow(4);
    odla_BindToOutputById((const odla_value_id) "Pow", out_Pow.data(), ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);
    std::vector<float> expected = {16, 1, 0.0816327, 625};
    for (int i = 0; i < 4; i++) {
      CHECK_LT(abs(expected[i] - out_Pow[i]), TOLLERANCE);
    }
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
  }

  SUBCASE("MATH OPS RECIPROCAL TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem(comp);

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

    std::vector<float> out_Reciprocal(4);
    odla_BindToOutputById((const odla_value_id) "Reciprocal",
                          out_Reciprocal.data(), ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);
    std::vector<float> expected = {-5, -0.25, 0.125, 0.111111};
    for (int i = 0; i < 4; i++) {
      CHECK_LT(abs(expected[i] - out_Reciprocal[i]), TOLLERANCE);
    }

    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
  }

  SUBCASE("MATH OPS REDUCEMAX TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem(comp);

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

    std::vector<float> out_ReduceMax(4);
    odla_BindToOutputById((const odla_value_id) "ReduceMax",
                          out_ReduceMax.data(), ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);
    std::vector<float> expected = {-0.2, 9, 0, 0};
    CHECK_EQ(expected, out_ReduceMax);
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
  }

  SUBCASE("MATH OPS REDUCEMIN TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem(comp);

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

    std::vector<float> out_ReduceMin(4);
    odla_BindToOutputById((const odla_value_id) "ReduceMin",
                          out_ReduceMin.data(), ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);
    std::vector<float> expected = {-4, 8, 0, 0};
    CHECK_EQ(expected, out_ReduceMin);
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
    set_computationItem(comp);

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

    std::vector<float> out_ReduceProd(4);
    odla_BindToOutputById((const odla_value_id) "ReduceProd",
                          out_ReduceProd.data(), ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);
    std::vector<float> expected = {-1.6, -36, 0, 0};
    CHECK_EQ(expected, out_ReduceProd);
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
  }

  SUBCASE("MATH OPS REDUCESUM TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem(comp);

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

    std::vector<float> out_ReduceSum(4);
    odla_BindToOutputById((const odla_value_id) "ReduceSum",
                          out_ReduceSum.data(), ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);
    std::vector<float> expected = {7.8, 5, 0, 0};
    CHECK_EQ(expected, out_ReduceSum);
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
  }

  SUBCASE("MATH OPS SIGN TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem(comp);

    auto input =
        odla_CreateArgument({ODLA_FLOAT32, {.size = 2, .dims = {2, 2}}},
                            (const odla_value_id)("input"));

    auto Sign = odla_Sign(input, (const odla_value_id) "Sign");
    odla_SetValueAsOutput(Sign);

    static odla_context ctx;
    odla_CreateContext(&ctx);

    float input_data[2 * 2] = {-0.2, -0.3, 1, 0.5};
    odla_BindToArgumentById((const odla_value_id) "input", input_data, ctx);

    std::vector<float> out_Sign(4);
    odla_BindToOutputById((const odla_value_id) "Sign", out_Sign.data(), ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);
    std::vector<float> expected = {-1, -1, 1, 1};
    CHECK_EQ(expected, out_Sign);

    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
  }

  SUBCASE("MATH OPS AND TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem(comp);

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

    bool gold[2 * 2] = {false, false, true, false};

    bool out_And[2 * 2] = {false, false, false, false};
    odla_BindToOutputById((const odla_value_id) "And", out_And, ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);
    for (int i = 0; i < 4; i++) {
      CHECK_EQ(out_And[i], gold[i]);
    }
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
  }

  SUBCASE("MATH OPS OR TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem(comp);

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

    bool gold[2 * 2] = {true, false, true, true};

    bool out_Or[2 * 2] = {false, false, false, false};
    odla_BindToOutputById((const odla_value_id) "Or", out_Or, ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

    for (int i = 0; i < 4; i++) {
      CHECK_EQ(out_Or[i], gold[i]);
    }
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
  }

  SUBCASE("MATH OPS NOT EQUAL TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem(comp);

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

    bool gold[2 * 2] = {true, false, false, true};

    bool out_Equal[2 * 2] = {false, false, false, false};
    odla_BindToOutputById((const odla_value_id) "NotEqual", out_Equal, ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

    for (int i = 0; i < 4; i++) {
      CHECK_EQ(out_Equal[i], gold[i]);
    }
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
  }

  SUBCASE("MATH OPS SUB TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem(comp);

    std::vector<float> c6 = {6.f};

    auto input =
        odla_CreateArgument({ODLA_FLOAT32, {.size = 1, .dims = {1, 1}}},
                            (const odla_value_id)("input"));

    auto const_6 =
        odla_CreateConstant({ODLA_FLOAT32, {.size = 1, .dims = {1, 1}}},
                            c6.data(), (const odla_value_id) "const");

    auto Sub_value = odla_Sub(input, const_6, (const odla_value_id) "Sub");

    odla_SetValueAsOutput(Sub_value);
    static odla_context ctx;
    odla_CreateContext(&ctx);
    std::vector<float> input_data = {2.0};
    odla_BindToArgumentById((const odla_value_id) "input", input_data.data(),
                            ctx);

    std::vector<float> out_sub = {0.f};
    odla_BindToOutputById((const odla_value_id) "Sub", out_sub.data(), ctx);
    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

    std::vector<float> expected = {-4.f};
    CHECK_EQ(expected, out_sub);

    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
  }

  SUBCASE("MATH OPS DIV TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem(comp);

    std::vector<float> c6 = {6.f};

    auto input =
        odla_CreateArgument({ODLA_FLOAT32, {.size = 1, .dims = {1, 1}}},
                            (const odla_value_id)("input"));

    auto const_6 =
        odla_CreateConstant({ODLA_FLOAT32, {.size = 1, .dims = {1, 1}}},
                            c6.data(), (const odla_value_id) "const");

    auto Div_value = odla_Div(input, const_6, (const odla_value_id) "Div");

    odla_SetValueAsOutput(Div_value);
    static odla_context ctx;
    odla_CreateContext(&ctx);
    std::vector<float> input_data = {12.0};
    odla_BindToArgumentById((const odla_value_id) "input", input_data.data(),
                            ctx);

    std::vector<float> out_div = {0.f};
    odla_BindToOutputById((const odla_value_id) "Div", out_div.data(), ctx);
    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

    std::vector<float> expected = {2.f};
    CHECK_EQ(expected, out_div);

    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
  }

  SUBCASE("MATH OPS ERF TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem(comp);

    std::vector<float> c6 = {6.f};

    auto input =
        odla_CreateArgument({ODLA_FLOAT32, {.size = 1, .dims = {1, 1}}},
                            (const odla_value_id)("input"));

    auto Erf_value = odla_Erf(input, (const odla_value_id) "Erf");

    odla_SetValueAsOutput(Erf_value);
    static odla_context ctx;
    odla_CreateContext(&ctx);
    std::vector<float> input_data = {12.0};
    odla_BindToArgumentById((const odla_value_id) "input", input_data.data(),
                            ctx);

    std::vector<float> out_erf = {0.f};
    odla_BindToOutputById((const odla_value_id) "Erf", out_erf.data(), ctx);
    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

    std::vector<float> expected = {1.f};

    CHECK_EQ(expected, out_erf);

    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
  }
}
