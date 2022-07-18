//===- odla_ops_nn_test.cc
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

TEST_CASE("NN OPS TESTING") {
  SUBCASE("AVERAGEPOOL OPS TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem(comp);

    auto input =
        odla_CreateArgument({ODLA_FLOAT32, {.size = 4, .dims = {1, 1, 4, 4}}},
                            (const odla_value_id)("input"));

    odla_memory_layout unused_layout;
    odla_uint32 dims[2] = {3, 3};
    odla_uint32 padding_front[2] = {0, 0};
    odla_uint32 padding_back[2] = {0, 0};
    odla_uint32 strides[2] = {1, 1};
    odla_value_shape output_dims;
    auto AveragePool = odla_AveragePool(
        input, unused_layout, dims, strides, padding_front, padding_back,
        output_dims, (const odla_value_id) "AveragePool");
    odla_SetValueAsOutput(AveragePool);

    static odla_context ctx;
    odla_CreateContext(&ctx);

    std::vector<float> input_data = {1, 2,  3,  4,  5,  6,  7,  8,
                                     9, 10, 11, 12, 13, 14, 15, 16};
    odla_BindToArgumentById((const odla_value_id) "input", input_data.data(),
                            ctx);

    std::vector<float> out_AveragePool(4);
    odla_BindToOutputById((const odla_value_id) "AveragePool",
                          out_AveragePool.data(), ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

    std::vector<float> expected = {6, 7, 10, 11};
    for (int i = 0; i < 4; i++) {
      CHECK_LT(abs(expected[i] - out_AveragePool[i]), TOLLERANCE);
    }
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
  }

  SUBCASE("BATCHNORMALIZATION OPS TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem(comp);

    auto input =
        odla_CreateArgument({ODLA_FLOAT32, {.size = 4, .dims = {1, 2, 1, 3}}},
                            (const odla_value_id)("input"));

    auto scale = odla_CreateArgument({ODLA_FLOAT32, {.size = 1, .dims = {2}}},
                                     (const odla_value_id)("scale"));

    auto offset = odla_CreateArgument({ODLA_FLOAT32, {.size = 1, .dims = {2}}},
                                      (const odla_value_id)("offset"));

    auto mean = odla_CreateArgument({ODLA_FLOAT32, {.size = 1, .dims = {2}}},
                                    (const odla_value_id)("mean"));

    auto var = odla_CreateArgument({ODLA_FLOAT32, {.size = 1, .dims = {2}}},
                                   (const odla_value_id)("var"));

    odla_memory_layout unused_layout;
    odla_value_shape output_dims;
    float epsilon = 1e-5;
    float scalar_scale = 1;
    float scalar_offset = 1;
    auto BatchNormalization = odla_BatchNormalization(
        input, unused_layout, mean, var, epsilon, scale, offset, scalar_scale,
        scalar_offset, (const odla_value_id) "BatchNormalization");
    odla_SetValueAsOutput(BatchNormalization);

    static odla_context ctx;
    odla_CreateContext(&ctx);

    std::vector<float> input_data = {-1, 0, 1, 2, 3, 4};
    odla_BindToArgumentById((const odla_value_id) "input", input_data.data(),
                            ctx);

    std::vector<float> scale_data = {1.0, 1.5};
    odla_BindToArgumentById((const odla_value_id) "scale", scale_data.data(),
                            ctx);

    std::vector<float> offset_data = {0, 1};
    odla_BindToArgumentById((const odla_value_id) "offset", offset_data.data(),
                            ctx);

    std::vector<float> mean_data = {0, 3};
    odla_BindToArgumentById((const odla_value_id) "mean", mean_data.data(),
                            ctx);

    std::vector<float> var_data = {1, 1.5};
    odla_BindToArgumentById((const odla_value_id) "var", var_data.data(), ctx);

    std::vector<float> out_BatchNormalization(6);
    odla_BindToOutputById((const odla_value_id) "BatchNormalization",
                          out_BatchNormalization.data(), ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);
    std::vector<float> expected = {-0.999995, 0, 0.999995,
                                   -0.224741, 1, 2.22474};
    for (int i = 0; i < 6; i++) {
      CHECK_LT(abs(expected[i] - out_BatchNormalization[i]), TOLLERANCE);
    }
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
  }

  SUBCASE("CONV OPS TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem(comp);

    auto input =
        odla_CreateArgument({ODLA_FLOAT32, {.size = 4, .dims = {1, 1, 5, 5}}},
                            (const odla_value_id)("input"));

    auto kernel =
        odla_CreateArgument({ODLA_FLOAT32, {.size = 4, .dims = {1, 1, 3, 3}}},
                            (const odla_value_id)("kernel"));

    odla_memory_layout unused_layout;
    odla_uint32 padding_front[2] = {1, 1};
    odla_uint32 padding_back[2] = {1, 1};
    odla_uint32 strides[2] = {1, 1};
    odla_uint32 dilations[2] = {1, 1};
    odla_value_shape output_dims;
    auto Conv = odla_Conv(input, unused_layout,
                          1, // group
                          kernel, unused_layout, strides, dilations,
                          padding_front, padding_back,
                          NULL,        // bias, unused
                          output_dims, // unused
                          (const odla_value_id) "Conv");
    odla_SetValueAsOutput(Conv);

    static odla_context ctx;
    odla_CreateContext(&ctx);

    std::vector<float> input_data = {0,  1,  2,  3,  4,  5,  6,  7,  8,
                                     9,  10, 11, 12, 13, 14, 15, 16, 17,
                                     18, 19, 20, 21, 22, 23, 24};
    odla_BindToArgumentById((const odla_value_id) "input", input_data.data(),
                            ctx);

    std::vector<float> kernel_data = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    odla_BindToArgumentById((const odla_value_id) "kernel", kernel_data.data(),
                            ctx);

    std::vector<float> out_Conv(25);
    odla_BindToOutputById((const odla_value_id) "Conv", out_Conv.data(), ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

    std::vector<float> expected = {12,  21,  27, 33,  24,  33,  54, 63,  72,
                                   51,  63,  99, 108, 117, 81,  93, 144, 153,
                                   162, 111, 72, 111, 117, 123, 84};

    CHECK_EQ(expected, out_Conv);
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
  }

  SUBCASE("DECONV OPS TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem(comp);

    auto input =
        odla_CreateArgument({ODLA_FLOAT32, {.size = 4, .dims = {1, 1, 3, 3}}},
                            (const odla_value_id)("input"));

    auto kernel =
        odla_CreateArgument({ODLA_FLOAT32, {.size = 4, .dims = {1, 2, 3, 3}}},
                            (const odla_value_id)("kernel"));

    odla_memory_layout unused_layout;
    odla_uint32 padding_front[2] = {0, 0};
    odla_uint32 padding_back[2] = {0, 0};
    odla_uint32 strides[2] = {1, 1};
    odla_uint32 dilations[2] = {1, 1};
    odla_value_shape output_dims;
    odla_uint32 group = 1;
    auto DeConv = odla_DeConv(input, unused_layout,
                              group, // group
                              kernel, unused_layout, strides, dilations,
                              padding_front, padding_back,
                              NULL,        // bias, unused
                              output_dims, // unused
                              (const odla_value_id) "DeConv");
    odla_SetValueAsOutput(DeConv);

    static odla_context ctx;
    odla_CreateContext(&ctx);

    std::vector<float> input_data = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    odla_BindToArgumentById((const odla_value_id) "input", input_data.data(),
                            ctx);

    std::vector<float> kernel_data = {1, 1, 1, 1, 1, 1, 1, 1, 1,
                                      1, 1, 1, 1, 1, 1, 1, 1, 1};
    odla_BindToArgumentById((const odla_value_id) "kernel", kernel_data.data(),
                            ctx);

    std::vector<float> out_DeConv(50);
    odla_BindToOutputById((const odla_value_id) "DeConv", out_DeConv.data(),
                          ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);
    std::vector<float> expected = {
        0,  1,  3,  3,  2,  3,  8,  15, 12, 7,  9,  21, 36, 27, 15, 9,  20,
        33, 24, 13, 6,  13, 21, 15, 8,  0,  1,  3,  3,  2,  3,  8,  15, 12,
        7,  9,  21, 36, 27, 15, 9,  20, 33, 24, 13, 6,  13, 21, 15, 8};
    for (int i = 0; i < 50; i++) {
      CHECK_LT(abs(expected[i] - out_DeConv[i]), TOLLERANCE);
    }
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
  }

  SUBCASE("ELU OPS TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem(comp);

    auto input = odla_CreateArgument({ODLA_FLOAT32, {.size = 1, .dims = {3}}},
                                     (const odla_value_id)("input"));

    float alpha = 2.0;
    auto Elu = odla_Elu(input, alpha, (const odla_value_id) "Elu");
    odla_SetValueAsOutput(Elu);

    static odla_context ctx;
    odla_CreateContext(&ctx);

    float input_data[3] = {-1, 0, 1};
    odla_BindToArgumentById((const odla_value_id) "input", input_data, ctx);

    std::vector<float> out_Elu(3);
    odla_BindToOutputById((const odla_value_id) "Elu", out_Elu.data(), ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

    std::vector<float> expected = {-1.26424, 0, 1};
    for (int i = 0; i < 3; i++) {
      CHECK_LT(abs(expected[i] - out_Elu[i]), TOLLERANCE);
    }
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
  }

  SUBCASE("HARDSIGMOID OPS TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem(comp);

    auto input = odla_CreateArgument({ODLA_FLOAT32, {.size = 1, .dims = {3}}},
                                     (const odla_value_id)("input"));

    float alpha = 0.5;
    float beta = 0.6;
    auto HardSigmoid = odla_HardSigmoid(input, alpha, beta,
                                        (const odla_value_id) "HardSigmoid");
    odla_SetValueAsOutput(HardSigmoid);

    static odla_context ctx;
    odla_CreateContext(&ctx);

    float input_data[3] = {-1, 0, 1};
    odla_BindToArgumentById((const odla_value_id) "input", input_data, ctx);

    std::vector<float> out_HardSigmoid(3);
    odla_BindToOutputById((const odla_value_id) "HardSigmoid",
                          out_HardSigmoid.data(), ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

    std::vector<float> expected = {0.1, 0.6, 1};
    for (int i = 0; i < 3; i++) {
      CHECK_LT(abs(expected[i] - out_HardSigmoid[i]), TOLLERANCE);
    }
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
  }

  SUBCASE("INSTANCENORMALIZATION OPS TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem(comp);

    auto input =
        odla_CreateArgument({ODLA_FLOAT32, {.size = 4, .dims = {1, 2, 1, 3}}},
                            (const odla_value_id)("input"));

    auto scale = odla_CreateArgument({ODLA_FLOAT32, {.size = 1, .dims = {2}}},
                                     (const odla_value_id)("scale"));

    auto offset = odla_CreateArgument({ODLA_FLOAT32, {.size = 1, .dims = {2}}},
                                      (const odla_value_id)("offset"));

    auto mean = odla_CreateArgument({ODLA_FLOAT32, {.size = 1, .dims = {2}}},
                                    (const odla_value_id)("mean"));

    auto var = odla_CreateArgument({ODLA_FLOAT32, {.size = 1, .dims = {2}}},
                                   (const odla_value_id)("var"));

    odla_memory_layout unused_layout;
    odla_value_shape output_dims;
    float epsilon = 1e-5;
    float scalar_scale = 1;
    float scalar_offset = 1;
    // auto InstanceNormalization = odla_InstanceNormalization(
    //     input, unused_layout, mean, var, epsilon, scale, offset, scalar_scale,
    //     scalar_offset, (const odla_value_id) "InstanceNormalization");
    // odla_SetValueAsOutput(InstanceNormalization);

    static odla_context ctx;
    odla_CreateContext(&ctx);

    std::vector<float> input_data = {-1, 0, 1, 2, 3, 4};
    odla_BindToArgumentById((const odla_value_id) "input", input_data.data(),
                            ctx);

    std::vector<float> scale_data = {1.0, 1.5};
    odla_BindToArgumentById((const odla_value_id) "scale", scale_data.data(),
                            ctx);

    std::vector<float> offset_data = {0, 1};
    odla_BindToArgumentById((const odla_value_id) "offset", offset_data.data(),
                            ctx);

    // std::vector<float> mean_data = {0, 3};
    // odla_BindToArgumentById((const odla_value_id) "mean", mean_data.data(),
    // ctx);

    // std::vector<float> var_data = {1, 1.5};
    // odla_BindToArgumentById((const odla_value_id) "var", var_data.data(),
    // ctx);

    // std::vector<float> out_InstanceNormalization(6);
    // odla_BindToOutputById((const odla_value_id) "InstanceNormalization",
    //                       out_InstanceNormalization.data(), ctx);

    // odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

    std::vector<float> expected = {-1.22474, 0, 1.22474, -0.837103, 1, 2.8371};
    // for (int i = 0; i < 6; i++) {
    //   CHECK_LT(abs(expected[i] - out_InstanceNormalization[i]), TOLLERANCE);
    // }
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
  }

  SUBCASE("LEAKYRELU OPS TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem(comp);

    auto input = odla_CreateArgument({ODLA_FLOAT32, {.size = 1, .dims = {3}}},
                                     (const odla_value_id)("input"));

    float alpha = 0.1;
    auto LeakyRelu =
        odla_LeakyRelu(input, alpha, (const odla_value_id) "LeakyRelu");
    odla_SetValueAsOutput(LeakyRelu);

    static odla_context ctx;
    odla_CreateContext(&ctx);

    float input_data[3] = {-1, 0, 1};
    odla_BindToArgumentById((const odla_value_id) "input", input_data, ctx);

    std::vector<float> out_LeakyRelu(3);
    odla_BindToOutputById((const odla_value_id) "LeakyRelu",
                          out_LeakyRelu.data(), ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

    std::vector<float> expected = {-0.1, 0, 1};
    CHECK_EQ(expected, out_LeakyRelu);
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
  }

  SUBCASE("LOGSOFTMAX OPS TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem(comp);

    auto input =
        odla_CreateArgument({ODLA_FLOAT32, {.size = 2, .dims = {1, 3}}},
                            (const odla_value_id)("input"));

    int axis = 1;
    auto LogSoftmax =
        odla_LogSoftmax(input, axis, (const odla_value_id) "LogSoftmax");
    odla_SetValueAsOutput(LogSoftmax);

    static odla_context ctx;
    odla_CreateContext(&ctx);

    float input_data[3] = {-1, 0, 1};
    odla_BindToArgumentById((const odla_value_id) "input", input_data, ctx);

    std::vector<float> out_LogSoftmax(3);
    odla_BindToOutputById((const odla_value_id) "LogSoftmax",
                          out_LogSoftmax.data(), ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);
    std::vector<float> expected = {-2.40761, -1.40761, -0.407606};
    for (int i = 0; i < 3; i++) {
      CHECK_LT(abs(expected[i] - out_LogSoftmax[i]), TOLLERANCE);
    }
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
  }

  SUBCASE("LSTM OPS TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem(comp);

    auto input =
        odla_CreateArgument({ODLA_FLOAT32, {.size = 3, .dims = {1, 3, 2}}},
                            (const odla_value_id)("input"));

    auto W =
        odla_CreateArgument({ODLA_FLOAT32, {.size = 3, .dims = {1, 12, 2}}},
                            (const odla_value_id)("W"));

    auto R =
        odla_CreateArgument({ODLA_FLOAT32, {.size = 3, .dims = {1, 12, 3}}},
                            (const odla_value_id)("R"));

    auto B = odla_CreateArgument({ODLA_FLOAT32, {.size = 2, .dims = {1, 24}}},
                                 (const odla_value_id)("B"));
    int input_size = 2;
    int hidden_size = 3;
    float weight_scale = 0.1;
    int number_of_gates = 4;
    float seq_len = 100;
    odla_rnn_direction direction = ODLA_RNN_FORWARD;
    odla_rnn_outputs rnn_outputs = ODLA_RNN_NO_STATE;
    // auto LSTM = odla_LSTM(
    //     input,
    //     {.size = 3, .dims = {1, number_of_gates * hidden_size, input_size}}, W,
    //     R, B, seq_len, hidden_size, direction, rnn_outputs,
    //     (const odla_value_id) "LSTM");

    // odla_SetValueAsOutput(LSTM.values[0]);
    // odla_SetValuesAsOutput(LSTM);
    static odla_context ctx;
    odla_CreateContext(&ctx);

    std::vector<float> input_data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    odla_BindToArgumentById((const odla_value_id) "input", input_data.data(),
                            ctx);

    std::vector<float> W_data(24, 0.1);
    odla_BindToArgumentById((const odla_value_id) "W", W_data.data(), ctx);

    std::vector<float> R_data(36, 0.1);
    odla_BindToArgumentById((const odla_value_id) "R", R_data.data(), ctx);

    std::vector<float> B_data(24, 0);
    odla_BindToArgumentById((const odla_value_id) "B", B_data.data(), ctx);

    // std::vector<float> out_LSTM(9);
    // odla_BindToOutputById((const odla_value_id) "LSTM0", out_LSTM.data(), ctx);

    // odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);
    std::vector<float> expected = {0.0952412, 0.0952412, 0.0952412,
                                   0.256064,  0.256064,  0.256064,
                                   0.403238,  0.403238,  0.403238};
    // for (int i = 0; i < 9; i++) {
    //   CHECK_LT(abs(expected[i] - out_LSTM[i]), TOLLERANCE);
    // }

    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
  }

  SUBCASE("MAXPOOL OPS TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem(comp);

    auto input =
        odla_CreateArgument({ODLA_FLOAT32, {.size = 4, .dims = {1, 1, 4, 4}}},
                            (const odla_value_id)("input"));

    odla_memory_layout unused_layout;
    odla_uint32 dims[2] = {3, 3};
    odla_uint32 padding_front[2] = {0, 0};
    odla_uint32 padding_back[2] = {0, 0};
    odla_uint32 strides[2] = {1, 1};
    odla_value_shape output_dims;
    auto MaxPool = odla_MaxPool(input, unused_layout, dims, strides,
                                padding_front, padding_back, output_dims,
                                (const odla_value_id) "MaxPool");
    odla_SetValueAsOutput(MaxPool);

    static odla_context ctx;
    odla_CreateContext(&ctx);

    std::vector<float> input_data = {1, 2,  3,  4,  5,  6,  7,  8,
                                     9, 10, 11, 12, 13, 14, 15, 16};
    odla_BindToArgumentById((const odla_value_id) "input", input_data.data(),
                            ctx);

    std::vector<float> out_MaxPool(4);
    odla_BindToOutputById((const odla_value_id) "MaxPool", out_MaxPool.data(),
                          ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);
    std::vector<float> expected = {11, 12, 15, 16};
    CHECK_EQ(expected, out_MaxPool);
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
  }

  SUBCASE("SELU OPS TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem(comp);

    auto input = odla_CreateArgument({ODLA_FLOAT32, {.size = 1, .dims = {3}}},
                                     (const odla_value_id)("input"));
    float alpha = 2;
    float gamma = 3;
    auto Selu = odla_Selu(input, alpha, gamma, (const odla_value_id) "Selu");
    odla_SetValueAsOutput(Selu);

    static odla_context ctx;
    odla_CreateContext(&ctx);

    float input_data[3] = {-1, 0, 1};
    odla_BindToArgumentById((const odla_value_id) "input", input_data, ctx);

    std::vector<float> out_Selu(3);
    odla_BindToOutputById((const odla_value_id) "Selu", out_Selu.data(), ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

    std::vector<float> expected = {-3.79272, 0, 3};
    for (int i = 0; i < 3; i++) {
      CHECK_LT(abs(expected[i] - out_Selu[i]), TOLLERANCE);
    }
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
  }

  SUBCASE("SIGMOID OPS TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem(comp);

    auto input = odla_CreateArgument({ODLA_FLOAT32, {.size = 1, .dims = {3}}},
                                     (const odla_value_id)("input"));

    auto Sigmoid = odla_Sigmoid(input, (const odla_value_id) "Sigmoid");
    odla_SetValueAsOutput(Sigmoid);

    static odla_context ctx;
    odla_CreateContext(&ctx);

    float input_data[3] = {-1, 0, 1};
    odla_BindToArgumentById((const odla_value_id) "input", input_data, ctx);

    std::vector<float> out_Sigmoid(3);
    odla_BindToOutputById((const odla_value_id) "Sigmoid", out_Sigmoid.data(),
                          ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

    std::vector<float> expected = {0.268941, 0.5, 0.731059};
    for (int i = 0; i < 3; i++) {
      CHECK_LT(abs(expected[i] - out_Sigmoid[i]), TOLLERANCE);
    }
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
  }

  SUBCASE("TANH OPS TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem(comp);

    auto input = odla_CreateArgument({ODLA_FLOAT32, {.size = 1, .dims = {3}}},
                                     (const odla_value_id)("input"));

    auto Tanh = odla_Tanh(input, (const odla_value_id) "Tanh");
    odla_SetValueAsOutput(Tanh);

    static odla_context ctx;
    odla_CreateContext(&ctx);

    float input_data[3] = {-1, 0, 1};
    odla_BindToArgumentById((const odla_value_id) "input", input_data, ctx);

    std::vector<float> out_Tanh(3);
    odla_BindToOutputById((const odla_value_id) "Tanh", out_Tanh.data(), ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

    std::vector<float> expected = {-0.761594, 0, 0.761594};
    for (int i = 0; i < 3; i++) {
      CHECK_LT(abs(expected[i] - out_Tanh[i]), TOLLERANCE);
    }

    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
  }

  SUBCASE("TOPK OPS TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem(comp, 1);

    auto input =
        odla_CreateArgument({ODLA_FLOAT32, {.size = 2, .dims = {3, 4}}},
                            (const odla_value_id)("input"));

    uint32_t axis = 1;
    uint32_t k = 1;
    odla_bool largest = true;
    odla_bool sorted = false;
    odla_value_type output_type;
    // auto Topk = odla_TopK(input, k, largest, sorted, axis, output_type,
    //                       (const odla_value_id) "Topk"); //todo change header
    // odla_SetValueAsOutput(Topk);

    static odla_context ctx;
    odla_CreateContext(&ctx);

    float input_data[3 * 4] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    odla_BindToArgumentById((const odla_value_id) "input", input_data, ctx);

    // todo change topk header
    // float out_Topk[3 * 4] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    // odla_BindToOutputById((const odla_value_id) "Topk", out_Topk, ctx);
    // odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
  }
}
