//===- odla_ops_test.cc
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

using namespace std;

TEST_CASE("GEMM OPS TESTING") {
  SUBCASE("GEMM OPS TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem(comp);

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
    set_computationItem(comp);

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

TEST_CASE("OPS TESTING") {
  SUBCASE("OPS Sub TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem(comp);

    auto lhs = odla_CreateArgument({ODLA_FLOAT32, {.size = 2, .dims = {2, 2}}},
                                   (const odla_value_id)("lhs"));

    auto rhs = odla_CreateArgument({ODLA_FLOAT32, {.size = 2, .dims = {2, 2}}},
                                   (const odla_value_id)("rhs"));

    auto Sub = odla_Add(lhs, rhs, (const odla_value_id) "Sub");
    odla_SetValueAsOutput(Sub);

    static odla_context ctx;
    odla_CreateContext(&ctx);

    float lhs_data[2 * 2] = {2.0, 1.0, 3.5, 5.0};
    odla_BindToArgumentById((const odla_value_id) "lhs", lhs_data, ctx);

    float rhs_data[2 * 2] = {4.0, 0.9, -2.0, 4.0};
    odla_BindToArgumentById((const odla_value_id) "rhs", rhs_data, ctx);

    std::vector<float> out_Sub(4);

    odla_BindToOutputById((const odla_value_id) "Sub", out_Sub.data(), ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

    std::vector<float> expected = {6, 1.9, 1.5, 9};
    CHECK_EQ(expected, out_Sub);
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
  }

  SUBCASE("OPS Floor TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem(comp);

    auto input =
        odla_CreateArgument({ODLA_FLOAT32, {.size = 2, .dims = {2, 2}}},
                            (const odla_value_id)("input"));

    auto Neg = odla_Floor(input, (const odla_value_id) "Neg");
    odla_SetValueAsOutput(Neg);

    static odla_context ctx;
    odla_CreateContext(&ctx);

    float input_data[2 * 2] = {2.0, 1.0, 3.5, 5.0};
    odla_BindToArgumentById((const odla_value_id) "input", input_data, ctx);

    std::vector<float> out_Neg(4);
    odla_BindToOutputById((const odla_value_id) "Neg", out_Neg.data(), ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);
    std::vector<float> expected = {2, 1, 3, 5};
    CHECK_EQ(expected, out_Neg);
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
  }

  SUBCASE("OPS Sqrt TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem(comp);

    auto input =
        odla_CreateArgument({ODLA_FLOAT32, {.size = 2, .dims = {2, 2}}},
                            (const odla_value_id)("input"));

    auto Neg = odla_Sqrt(input, (const odla_value_id) "Neg");
    odla_SetValueAsOutput(Neg);

    static odla_context ctx;
    odla_CreateContext(&ctx);

    float input_data[2 * 2] = {2.0, 1.0, 3.5, 5.0};
    odla_BindToArgumentById((const odla_value_id) "input", input_data, ctx);

    std::vector<float> out_Neg(4);
    odla_BindToOutputById((const odla_value_id) "Neg", out_Neg.data(), ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

    std::vector<float> expected = {2, 1, 3.5, 5};
    CHECK_EQ(expected, expected);
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
  }

  SUBCASE("OPS Rsqrt TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem(comp);

    auto input =
        odla_CreateArgument({ODLA_FLOAT32, {.size = 2, .dims = {2, 2}}},
                            (const odla_value_id)("input"));

    auto Neg = odla_Rsqrt(input, (const odla_value_id) "Neg");
    odla_SetValueAsOutput(Neg);

    static odla_context ctx;
    odla_CreateContext(&ctx);

    float input_data[2 * 2] = {2.0, 1.0, 3.5, 5.0};
    odla_BindToArgumentById((const odla_value_id) "input", input_data, ctx);

    std::vector<float> out_Neg(4);
    odla_BindToOutputById((const odla_value_id) "Neg", out_Neg.data(), ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

    std::vector<float> expected = {2, 1, 3.5, 5};
    CHECK_EQ(expected, expected);
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
  }

  SUBCASE("OPS Relu TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem(comp);

    auto input =
        odla_CreateArgument({ODLA_FLOAT32, {.size = 2, .dims = {2, 2}}},
                            (const odla_value_id)("input"));

    auto Neg = odla_Relu(input, (const odla_value_id) "Neg");
    odla_SetValueAsOutput(Neg);

    static odla_context ctx;
    odla_CreateContext(&ctx);

    float input_data[2 * 2] = {2.0, 1.0, 3.5, 5.0};
    odla_BindToArgumentById((const odla_value_id) "input", input_data, ctx);

    std::vector<float> out_Neg(4);
    odla_BindToOutputById((const odla_value_id) "Neg", out_Neg.data(), ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

    std::vector<float> expected = {2, 1, 3.5, 5};
    CHECK_EQ(expected, out_Neg);
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
  }
}