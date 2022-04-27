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

TEST_CASE("PROCESS OPS TESTING") {

   SUBCASE("CAST OPS TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem();


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

    int32_t out_Cast[4] = {0, 0, 0, 0};
    odla_BindToOutputById((const odla_value_id) "Cast", out_Cast, ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

    std::cout << "out_Cast = [";
    for (int i = 0; i < 4; i++) {
      std::cout << out_Cast[i] << ", ";
    }
    std::cout << "]" << std::endl;

    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
  }
   
   SUBCASE("CONCAT OPS TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem();


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

    std::cout << "out_Concat = [";
    for (int i = 0; i < 14; i++) {
      std::cout << out_Concat[i] << ", ";
    }
    std::cout << "]" << std::endl;
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);

  }

   SUBCASE("EXPANDDIM OPS TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem();


    auto input = odla_CreateArgument({ODLA_FLOAT32, {.size = 1, .dims = {1}}},
                                     (const odla_value_id)("input"));

    int axis = 1;
    odla_value_shape output_shape{.size = 2, .dims = {1, 1}};
    //   odla_value_shape output_shape{.size=3, .dims={2, 1, 6}};
    auto ExpandDim = odla_ExpandDims(input, output_shape,
                                     (const odla_value_id) "ExpandDim");
    odla_SetValueAsOutput(ExpandDim);

    static odla_context ctx;
    odla_CreateContext(&ctx);

    std::vector<float> input_data = {1.0};
    //   std::vector<float> input_data = {1, 2, 3};
    odla_BindToArgumentById((const odla_value_id) "input", input_data.data(),
                            ctx);

    float out_ExpandDim[14] = {0};
    odla_BindToOutputById((const odla_value_id) "ExpandDim", out_ExpandDim,
                          ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

    auto shape = comp->builder->getTensorShape(ExpandDim->tensor_id);
    std::cout << "result shape:[";
    for (int i = 0; i < shape.size(); ++i) {
      std::cout << shape[i] << ", ";
    }
    std::cout << "]" << std::endl;
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);

    //   std::cout << "out_ExpandDim = [";
    //   for (int i = 0; i < 14; i++) {
    //     std::cout << out_ExpandDim[i] << ", ";
    //   }
    //   std::cout << "]" << std::endl;
  }
   
   SUBCASE("PAD OPS TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem();


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

    float out_Pad[16] = {0};
    odla_BindToOutputById((const odla_value_id) "Pad", out_Pad, ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

    std::cout << "out_Pad = [";
    for (int i = 0; i < 16; i++) {
      std::cout << out_Pad[i] << ", ";
    }
    std::cout << "]" << std::endl;
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);

  }
   
   SUBCASE("RESIZE OPS TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem();


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

    float out_Resize[16] = {0};
    odla_BindToOutputById((const odla_value_id) "Resize", out_Resize, ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

    std::cout << "out_Resize = [";
    for (int i = 0; i < 16; i++) {
      std::cout << out_Resize[i] << ", ";
    }
    std::cout << "]" << std::endl;

    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);

  }
   
   SUBCASE("SHAPE OPS TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem();

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

    float out_Shape[16] = {0};
    odla_BindToOutputById((const odla_value_id) "Reshape", out_Shape, ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);
    auto shape = comp->builder->getTensorShape(Reshape->tensor_id);

    // auto size = Reshape->tensor_id;
    std::cout << " shape:[";
    for (int i = 0; i < shape.size(); ++i) {
      std::cout << shape[i] << ", ";
    }
    std::cout << "]" << std::endl;
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
  }
   
   SUBCASE("SQUEEZE OPS TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem();


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

   SUBCASE("SHAPE OPS TEST"){}
  
   SUBCASE("TILE  TEST") {
      odla_computation comp;
      CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
      set_computationItem(comp, 1);

      auto input =
          odla_CreateArgument({ODLA_FLOAT32, {.size = 2, .dims = {2, 3}}},
                              (const odla_value_id)("input"));

      odla_value_shape output_dims = {.size = 2, .dims = {4, 3}};
      const odla_uint32* rep = new  odla_uint32[2]{3,2};

      auto Tile =
          odla_Tile(input, rep, output_dims, (const odla_value_id) "Tile");

      odla_SetValueAsOutput(Tile);
      static odla_context ctx;
      odla_CreateContext(&ctx);

      std::vector<float> input_data = {1, 2, 3, 4, 5, 6};
      odla_BindToArgumentById((const odla_value_id) "input", input_data.data(),
                              ctx);

      float out_Shape[16] = {0};
      odla_BindToOutputById((const odla_value_id) "Tile", out_Shape, ctx);

      odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);
      auto shape = comp->builder->getTensorShape(Tile->tensor_id);

      // auto size = Tile->tensor_id;
      std::cout << " shape:[";
      for (int i = 0; i < shape.size(); ++i) {
         CHECK_EQ(shape[i], rep[i] * comp->builder->getTensorShape(input->tensor_id)[i]);
        //std::cout << shape[i] << ", ";
      }
      std::cout << "]" << std::endl;
      odla_DestroyComputation(comp);
      odla_DestroyContext(ctx);
    }




}

