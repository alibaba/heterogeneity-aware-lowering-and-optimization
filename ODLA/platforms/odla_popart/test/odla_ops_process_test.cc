//===- odla_ops_process_test.cc
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

using namespace std;

TEST_CASE("PROCESS OPS TESTING") {
  SUBCASE("CAST OPS TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem(comp);

    auto input = odla_CreateArgument({ODLA_FLOAT32, {.size = 1, .dims = {4}}},
                                     (const odla_value_id)("input"));

    odla_element_type element_type = ODLA_INT32;
    auto AveragePool =
        odla_Cast(input, element_type, (const odla_value_id) "Cast");
    odla_SetValueAsOutput(AveragePool);

    static odla_context ctx;
    odla_CreateContext(&ctx);

    std::vector<float> input_data = {1.2, 2.3, 3.4, 4.5};
    odla_BindToArgumentById((const odla_value_id) "input", input_data.data(),
                            ctx);

    std::vector<int32_t> out_Cast(4);
    odla_BindToOutputById((const odla_value_id) "Cast", out_Cast.data(), ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

    std::vector<int32_t> expected = {1, 2, 3, 4};
    CHECK_EQ(expected, out_Cast);

    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
  }

  SUBCASE("CONCAT OPS TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem(comp);

    auto input_1 =
        odla_CreateArgument({ODLA_FLOAT32, {.size = 2, .dims = {2, 1}}},
                            (const odla_value_id)("input_1"));

    auto input_2 =
        odla_CreateArgument({ODLA_FLOAT32, {.size = 2, .dims = {2, 2}}},
                            (const odla_value_id)("input_2"));

    auto input_3 =
        odla_CreateArgument({ODLA_FLOAT32, {.size = 2, .dims = {2, 4}}},
                            (const odla_value_id)("input_3"));

    int axis = 1;
    odla_value_shape output_shape;
    auto Concat =
        odla_Concat({.size = 3, .values = {input_1, input_2, input_3}}, axis,
                    output_shape, (const odla_value_id) "Concat");
    odla_SetValueAsOutput(Concat);

    static odla_context ctx;
    odla_CreateContext(&ctx);

    std::vector<float> input_data_1 = {5, 8};
    odla_BindToArgumentById((const odla_value_id) "input_1",
                            input_data_1.data(), ctx);

    std::vector<float> input_data_2 = {1, 3, 4, 7};
    odla_BindToArgumentById((const odla_value_id) "input_2",
                            input_data_2.data(), ctx);

    std::vector<float> input_data_3 = {1, 2, 3, 5, 7, 8, 9, 0};
    odla_BindToArgumentById((const odla_value_id) "input_3",
                            input_data_3.data(), ctx);

    float out_Concat[14] = {0};
    odla_BindToOutputById((const odla_value_id) "Concat", out_Concat, ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

    std::vector<float> expected = {
        5, 1, 3, 1, 2, 3, 5, 8, 4, 7, 7, 8, 9, 0,
    };
    for (int i = 0; i < 14; i++) {
      CHECK_EQ(expected[i], out_Concat[i]);
    }
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
  }

  SUBCASE("EXPANDDIM OPS TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem(comp);

    auto input = odla_CreateArgument({ODLA_FLOAT32, {.size = 1, .dims = {1}}},
                                     (const odla_value_id)("input"));

    int axis = 1;
    odla_value_shape output_shape{.size = 2, .dims = {1, 1}};
    //   odla_value_shape output_shape{.size=3, .dims={2, 1, 6}};
    auto ExpandDim =
        odla_ExpandDims(input, output_shape, (const odla_value_id) "ExpandDim");
    odla_SetValueAsOutput(ExpandDim);

    static odla_context ctx;
    odla_CreateContext(&ctx);

    std::vector<float> input_data = {1.0};
    //   std::vector<float> input_data = {1, 2, 3};
    odla_BindToArgumentById((const odla_value_id) "input", input_data.data(),
                            ctx);

    std::vector<float> out_ExpandDim(4);
    odla_BindToOutputById((const odla_value_id) "ExpandDim",
                          out_ExpandDim.data(), ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

    auto shape = comp->builder->getTensorShape(ExpandDim->tensor_id);
    std::cout << "result shape:[";
    for (int i = 0; i < shape.size(); ++i) {
      std::cout << shape[i] << ", ";
    }
    std::cout << "]" << std::endl;
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
  }

  SUBCASE("PAD OPS TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem(comp);

    auto input =
        odla_CreateArgument({ODLA_FLOAT32, {.size = 2, .dims = {2, 2}}},
                            (const odla_value_id)("input"));

    odla_uint32 padding_front[2] = {1, 1};
    odla_uint32 padding_back[2] = {1, 1};
    odla_value_shape output_dims;
    auto Pad = odla_Pad(input, padding_front, padding_back, output_dims,
                        (const odla_value_id) "Pad");
    odla_SetValueAsOutput(Pad);

    static odla_context ctx;
    odla_CreateContext(&ctx);

    std::vector<float> input_data = {1, 2, 3, 4};
    odla_BindToArgumentById((const odla_value_id) "input", input_data.data(),
                            ctx);

    std::vector<float> out_Pad(16);
    odla_BindToOutputById((const odla_value_id) "Pad", out_Pad.data(), ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);
    std::vector<float> expected = {0, 0, 0, 0, 0, 1, 2, 0,
                                   0, 3, 4, 0, 0, 0, 0, 0};
    CHECK_EQ(expected, out_Pad);
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
  }

  SUBCASE("RESIZE OPS TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem(comp);

    auto input =
        odla_CreateArgument({ODLA_FLOAT32, {.size = 4, .dims = {1, 1, 2, 4}}},
                            (const odla_value_id)("input"));
    odla_interpolation_mode interpolation = ODLA_NEAREST;
    odla_resize_coordinate_mode mode;
    odla_uint32 axes_mask;
    odla_value_shape output_dims{.size = 4, .dims = {1, 1, 4, 2}};
    auto Resize = odla_Resize(input, interpolation, mode, axes_mask,
                              output_dims, (const odla_value_id) "Resize");
    odla_SetValueAsOutput(Resize);

    static odla_context ctx;
    odla_CreateContext(&ctx);

    std::vector<float> input_data = {1, 2, 3, 4, 5, 6, 7, 8};
    odla_BindToArgumentById((const odla_value_id) "input", input_data.data(),
                            ctx);

    std::vector<float> out_Resize(16);
    odla_BindToOutputById((const odla_value_id) "Resize", out_Resize.data(),
                          ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

    std::vector<float> expected = {1, 3, 1, 3, 5, 7, 5, 7,
                                   0, 0, 0, 0, 0, 0, 0, 0};
    CHECK_EQ(expected, out_Resize);

    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
  }

  SUBCASE("RESHAPE OPS TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem(comp);

    auto input =
        odla_CreateArgument({ODLA_FLOAT32, {.size = 2, .dims = {2, 3}}},
                            (const odla_value_id)("input"));

    odla_value_shape output_dims = {.size = 2, .dims = {3, 2}};
    auto Reshape =
        odla_Reshape(input, output_dims, (const odla_value_id) "Reshape");

    odla_SetValueAsOutput(Reshape);
    static odla_context ctx;
    odla_CreateContext(&ctx);

    std::vector<float> input_data = {1, 2, 3, 4, 5, 6};
    odla_BindToArgumentById((const odla_value_id) "input", input_data.data(),
                            ctx);

    std::vector<float> out_Shape(16);
    odla_BindToOutputById((const odla_value_id) "Reshape", out_Shape.data(),
                          ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);
    auto shape = comp->builder->getTensorShape(Reshape->tensor_id);

    // auto size = Reshape->tensor_id;
    float expected[2] = {3, 2};
    for (int i = 0; i < shape.size(); ++i) {
      CHECK_EQ(shape[i], shape[i]);
    }
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
  }

  SUBCASE("SHAPE OPS TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem(comp);

    auto input =
        odla_CreateArgument({ODLA_FLOAT32, {.size = 2, .dims = {2, 3}}},
                            (const odla_value_id)("input"));

    odla_value_shape output_dims = {.size = 2, .dims = {3, 2}};
    auto Shape = odla_Shape(input, output_dims, (const odla_value_id) "shape");

    odla_SetValueAsOutput(Shape);

    auto shape = comp->builder->getTensorShape(Shape->tensor_id);

    // auto size = shape->tensor_id;
    float expected[2] = {3, 2};
    for (int i = 0; i < shape.size(); ++i) {
      CHECK_EQ(shape[i], shape[i]);
    }
    odla_DestroyComputation(comp);
  }

  SUBCASE("TILE OPS TEST") {
    odla_computation comp;
    odla_CreateComputation(&comp);
    set_computationItem(comp);

    auto input =
        odla_CreateArgument({ODLA_FLOAT32, {.size = 2, .dims = {2, 2}}},
                            (const odla_value_id)("input"));

    odla_value_shape output_dims = {.size = 2, .dims = {2, 2}};
    odla_uint32 repeat_time = 1;
    auto Tile = odla_Tile(input, &repeat_time, output_dims,
                          (const odla_value_id) "Tile");
    //todo need to verify result
    odla_SetValueAsOutput(Tile);
    odla_DestroyComputation(comp);
  }

  SUBCASE("SQUEEZE OPS TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem(comp);

    auto input =
        odla_CreateArgument({ODLA_FLOAT32, {.size = 3, .dims = {2, 1, 3}}},
                            (const odla_value_id)("input"));
    uint32_t axes_squeeze_num = 1;
    uint32_t axes_squeeze[1] = {1};
    odla_value_shape output_dims;
    auto Squeeze = odla_Squeeze(input, axes_squeeze_num, axes_squeeze,
                                output_dims, (const odla_value_id) "Squeeze");
    odla_SetValueAsOutput(Squeeze);

    static odla_context ctx;
    odla_CreateContext(&ctx);

    std::vector<float> input_data = {1, 2, 3, 4, 5, 6};
    odla_BindToArgumentById((const odla_value_id) "input", input_data.data(),
                            ctx);

    float out_Squeeze[16] = {0};
    odla_BindToOutputById((const odla_value_id) "Squeeze", out_Squeeze, ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

    auto shape = comp->builder->getTensorShape(Squeeze->tensor_id);
    std::cout << "squeeze result shape:[";
    for (int i = 0; i < shape.size(); ++i) {
      std::cout << shape[i] << ", ";
    }
    std::cout << "]" << std::endl;
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
  }
}