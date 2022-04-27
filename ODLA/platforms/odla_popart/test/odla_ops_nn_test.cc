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

TEST_CASE("NN OPS TESTING") {
  SUBCASE("AVERAGEPOOL OPS TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem();

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

    float out_AveragePool[4] = {0, 0, 0, 0};
    odla_BindToOutputById((const odla_value_id) "AveragePool", out_AveragePool,
                          ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

    std::cout << "out_AveragePool = [";
    for (int i = 0; i < 4; i++) {
      std::cout << out_AveragePool[i] << ", ";
    }
    std::cout << "]" << std::endl;
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
  }

  SUBCASE("BATCHNORMALIZATION OPS TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem();

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

    float out_BatchNormalization[6] = {0, 0, 0, 0, 0, 0};
    odla_BindToOutputById((const odla_value_id) "BatchNormalization",
                          out_BatchNormalization, ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

    std::cout << "out_BatchNormalization = [";
    for (int i = 0; i < 6; i++) {
      std::cout << out_BatchNormalization[i] << ", ";
    }
    std::cout << "]" << std::endl;
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
  }

  SUBCASE("CONV OPS TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem();

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

    float out_Conv[25] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    odla_BindToOutputById((const odla_value_id) "Conv", out_Conv, ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

    std::cout << "out_Conv = [";
    for (int i = 0; i < 25; i++) {
      std::cout << out_Conv[i] << ", ";
    }
    std::cout << "]" << std::endl;
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
  }

  SUBCASE("DECONV OPS TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem();

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

    float out_DeConv[60] = {0};
    odla_BindToOutputById((const odla_value_id) "DeConv", out_DeConv, ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

    std::cout << "out_DeConv = [";
    for (int i = 0; i < 50; i++) {
      std::cout << out_DeConv[i] << ", ";
    }
    std::cout << "]" << std::endl;
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
  }

  SUBCASE("ELU OPS TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem();

    auto input = odla_CreateArgument({ODLA_FLOAT32, {.size = 1, .dims = {3}}},
                                     (const odla_value_id)("input"));

    float alpha = 2.0;
    auto Elu = odla_Elu(input, alpha, (const odla_value_id) "Elu");
    odla_SetValueAsOutput(Elu);

    static odla_context ctx;
    odla_CreateContext(&ctx);

    float input_data[3] = {-1, 0, 1};
    odla_BindToArgumentById((const odla_value_id) "input", input_data, ctx);

    float out_Elu[3] = {0, 0, 0};
    odla_BindToOutputById((const odla_value_id) "Elu", out_Elu, ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

    std::cout << "out_Elu = [";
    for (int i = 0; i < 3; i++) {
      std::cout << out_Elu[i] << ", ";
    }
    std::cout << "]" << std::endl;
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
  }

  SUBCASE("HARDSIGMOID OPS TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem();

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

    float out_HardSigmoid[3] = {0, 0, 0};
    odla_BindToOutputById((const odla_value_id) "HardSigmoid", out_HardSigmoid,
                          ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

    std::cout << "out_HardSigmoid = [";
    for (int i = 0; i < 3; i++) {
      std::cout << out_HardSigmoid[i] << ", ";
    }
    std::cout << "]" << std::endl;
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
  }

  SUBCASE("INSTANCENORMALIZATION OPS TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem();

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
    auto InstanceNormalization = odla_InstanceNormalization(
        input, unused_layout, mean, var, epsilon, scale, offset, scalar_scale,
        scalar_offset, (const odla_value_id) "InstanceNormalization");
    odla_SetValueAsOutput(InstanceNormalization);

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

    float out_InstanceNormalization[6] = {0, 0, 0, 0, 0, 0};
    odla_BindToOutputById((const odla_value_id) "InstanceNormalization",
                          out_InstanceNormalization, ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

    std::cout << "out_InstanceNormalization = [";
    for (int i = 0; i < 6; i++) {
      std::cout << out_InstanceNormalization[i] << ", ";
    }
    std::cout << "]" << std::endl;
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
  }

  SUBCASE("LEAKYRELU OPS TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem();

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

    float out_LeakyRelu[3] = {0, 0, 0};
    odla_BindToOutputById((const odla_value_id) "LeakyRelu", out_LeakyRelu,
                          ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

    std::cout << "out_LeakyRelu = [";
    for (int i = 0; i < 3; i++) {
      std::cout << out_LeakyRelu[i] << ", ";
    }
    std::cout << "]" << std::endl;
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
  }

  SUBCASE("LOGSOFTMAX OPS TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem();

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

    float out_LogSoftmax[3] = {0, 0, 0};
    odla_BindToOutputById((const odla_value_id) "LogSoftmax", out_LogSoftmax,
                          ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

    std::cout << "out_LogSoftmax = [";
    for (int i = 0; i < 3; i++) {
      std::cout << out_LogSoftmax[i] << ", ";
    }
    std::cout << "]" << std::endl;
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
  }

  SUBCASE("LSTM OPS TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem();

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
    auto LSTM = odla_LSTM(
        input,
        {.size = 3, .dims = {1, number_of_gates * hidden_size, input_size}}, W,
        R, B, seq_len, hidden_size, direction, rnn_outputs,
        (const odla_value_id) "LSTM");

    odla_SetValueAsOutput(LSTM.values[0]);
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

    // std::vector<float> out_LSTM(75, 0);
    float out_LSTM[9] = {0};
    odla_BindToOutputById((const odla_value_id) "LSTM0", out_LSTM, ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

      std::cout << "out_LSTM = [";
      for (int i = 0; i < 9; i++) {
        std::cout << out_LSTM[i] << ", ";
      }
      std::cout << "]" << std::endl;
      odla_DestroyComputation(comp);
      odla_DestroyContext(ctx);
    }
  
  SUBCASE("MAXPOOL OPS TEST") {
      odla_computation comp;
      CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
      set_computationItem();

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

      float out_MaxPool[4] = {0, 0, 0, 0};
      odla_BindToOutputById((const odla_value_id) "MaxPool", out_MaxPool, ctx);

      odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

      std::cout << "out_MaxPool = [";
      for (int i = 0; i < 4; i++) {
        std::cout << out_MaxPool[i] << ", ";
      }
      std::cout << "]" << std::endl;
      odla_DestroyComputation(comp);
      odla_DestroyContext(ctx);
    }
  
  SUBCASE("PRELU OPS TEST") {
      odla_computation comp;
      CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
      set_computationItem();

      auto input = odla_CreateArgument({ODLA_FLOAT32, {.size = 1, .dims = {3}}},
                                       (const odla_value_id)("input"));
      float sloap = 1;
      auto PRelu = odla_PRelu(input, sloap, (const odla_value_id)("PRelu"));
      odla_SetValueAsOutput(PRelu);

      static odla_context ctx;
      odla_CreateContext(&ctx);

      float input_data[3] = {-1, 0, 1};
      odla_BindToArgumentById((const odla_value_id) "input", input_data, ctx);

      float out_PRelu[3] = {0, 0, 0};
      odla_BindToOutputById((const odla_value_id) "PRelu", out_PRelu, ctx);

      odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

      std::cout << "out_PRelu = [";
      for (int i = 0; i < 3; i++) {
        std::cout << out_PRelu[i] << ", ";
      }
      std::cout << "]" << std::endl;
      odla_DestroyComputation(comp);
      odla_DestroyContext(ctx);
    }
  
  SUBCASE("SELU OPS TEST") {
      odla_computation comp;
      CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
      set_computationItem();

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

      float out_Selu[3] = {0, 0, 0};
      odla_BindToOutputById((const odla_value_id) "Selu", out_Selu, ctx);

      odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

      std::cout << "out_Selu = [";
      for (int i = 0; i < 3; i++) {
        std::cout << out_Selu[i] << ", ";
      }
      std::cout << "]" << std::endl;
      odla_DestroyComputation(comp);
      odla_DestroyContext(ctx);
    }
  
  SUBCASE("SIGMOID OPS TEST") {
      odla_computation comp;
      CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
      set_computationItem();

      auto input = odla_CreateArgument({ODLA_FLOAT32, {.size = 1, .dims = {3}}},
                                       (const odla_value_id)("input"));

      auto Sigmoid = odla_Sigmoid(input, (const odla_value_id) "Sigmoid");
      odla_SetValueAsOutput(Sigmoid);

      static odla_context ctx;
      odla_CreateContext(&ctx);

      float input_data[3] = {-1, 0, 1};
      odla_BindToArgumentById((const odla_value_id) "input", input_data, ctx);

      float out_Sigmoid[3] = {0, 0, 0};
      odla_BindToOutputById((const odla_value_id) "Sigmoid", out_Sigmoid, ctx);

      odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

      std::cout << "out_Sigmoid = [";
      for (int i = 0; i < 3; i++) {
        std::cout << out_Sigmoid[i] << ", ";
      }
      std::cout << "]" << std::endl;
      odla_DestroyComputation(comp);
      odla_DestroyContext(ctx);
    }
  
  SUBCASE("TANH OPS TEST") {
      odla_computation comp;
      CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
      set_computationItem();

      auto input = odla_CreateArgument({ODLA_FLOAT32, {.size = 1, .dims = {3}}},
                                       (const odla_value_id)("input"));

      auto Tanh = odla_Tanh(input, (const odla_value_id) "Tanh");
      odla_SetValueAsOutput(Tanh);

      static odla_context ctx;
      odla_CreateContext(&ctx);

      float input_data[3] = {-1, 0, 1};
      odla_BindToArgumentById((const odla_value_id) "input", input_data, ctx);

      float out_Tanh[3] = {0, 0, 0};
      odla_BindToOutputById((const odla_value_id) "Tanh", out_Tanh, ctx);

      odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

      std::cout << "out_Tanh = [";
      for (int i = 0; i < 3; i++) {
        std::cout << out_Tanh[i] << ", ";
      }
      std::cout << "]" << std::endl;
      odla_DestroyComputation(comp);
      odla_DestroyContext(ctx);
    }
  
  SUBCASE("TOPK OPS TEST") {
      odla_computation comp;
      CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
      set_computationItem();

      auto input =
          odla_CreateArgument({ODLA_FLOAT32, {.size = 2, .dims = {3, 4}}},
                              (const odla_value_id)("input"));

      uint32_t axis = 1;
      uint32_t k = 1;
      odla_bool largest = true;
      odla_bool sorted = false;
      odla_value_type output_type;
      auto Topk = odla_TopK(input, k, largest, sorted, axis, output_type,
                            (const odla_value_id) "Topk");
      odla_SetValueAsOutput(Topk);

      static odla_context ctx;
      odla_CreateContext(&ctx);

      float input_data[3 * 4] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
      odla_BindToArgumentById((const odla_value_id) "input", input_data, ctx);

      float out_Topk[3 * 4] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
      odla_BindToOutputById((const odla_value_id) "Topk", out_Topk, ctx);

      odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

      std::cout << "out_Topk = [";
      for (int i = 0; i < 3; i++) {
        std::cout << out_Topk[i] << ", ";
      }
      std::cout << "]" << std::endl;

      odla_DestroyComputation(comp);
      odla_DestroyContext(ctx);
    }
  
  SUBCASE("POSTPROCESS OPS TEST") {
     
    }
   
  }

