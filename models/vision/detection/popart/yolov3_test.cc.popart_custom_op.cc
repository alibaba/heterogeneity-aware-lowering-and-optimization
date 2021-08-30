//===- Halo Compiler Generated File --------------------------------===//

#include <ODLA/odla.h>
#include <stdio.h>
#include <time.h>

extern const float W2[255 * 1024 * 1 * 1];
extern const float W1[255 * 512 * 1 * 1];
extern const float W[255 * 256 * 1 * 1];
extern const float Resize_scales[4];
extern const float B2[255];
extern const float B1[255];
extern const float B[255];
extern const float batch_normalization_1_1[1 * 32 * 1 * 1];
extern const float W74[32 * 3 * 3 * 3];
extern const float batch_normalization_2_1[1 * 64 * 1 * 1];
extern const float W73[64 * 32 * 3 * 3];
extern const float batch_normalization_3_1[1 * 32 * 1 * 1];
extern const float W72[32 * 64 * 1 * 1];
extern const float batch_normalization_4_1[1 * 64 * 1 * 1];
extern const float W71[64 * 32 * 3 * 3];
extern const float batch_normalization_5_1[1 * 128 * 1 * 1];
extern const float W70[128 * 64 * 3 * 3];
extern const float batch_normalization_6_1[1 * 64 * 1 * 1];
extern const float W69[64 * 128 * 1 * 1];
extern const float batch_normalization_7_1[1 * 128 * 1 * 1];
extern const float W68[128 * 64 * 3 * 3];
extern const float batch_normalization_8_1[1 * 64 * 1 * 1];
extern const float W67[64 * 128 * 1 * 1];
extern const float batch_normalization_9_1[1 * 128 * 1 * 1];
extern const float W66[128 * 64 * 3 * 3];
extern const float batch_normalization_10_1[1 * 256 * 1 * 1];
extern const float W65[256 * 128 * 3 * 3];
extern const float batch_normalization_11_1[1 * 128 * 1 * 1];
extern const float W64[128 * 256 * 1 * 1];
extern const float batch_normalization_12_1[1 * 256 * 1 * 1];
extern const float W63[256 * 128 * 3 * 3];
extern const float batch_normalization_13_1[1 * 128 * 1 * 1];
extern const float W62[128 * 256 * 1 * 1];
extern const float batch_normalization_14_1[1 * 256 * 1 * 1];
extern const float W61[256 * 128 * 3 * 3];
extern const float batch_normalization_15_1[1 * 128 * 1 * 1];
extern const float W60[128 * 256 * 1 * 1];
extern const float batch_normalization_16_1[1 * 256 * 1 * 1];
extern const float W59[256 * 128 * 3 * 3];
extern const float batch_normalization_17_1[1 * 128 * 1 * 1];
extern const float W58[128 * 256 * 1 * 1];
extern const float batch_normalization_18_1[1 * 256 * 1 * 1];
extern const float W57[256 * 128 * 3 * 3];
extern const float batch_normalization_19_1[1 * 128 * 1 * 1];
extern const float W56[128 * 256 * 1 * 1];
extern const float batch_normalization_20_1[1 * 256 * 1 * 1];
extern const float W55[256 * 128 * 3 * 3];
extern const float batch_normalization_21_1[1 * 128 * 1 * 1];
extern const float W54[128 * 256 * 1 * 1];
extern const float batch_normalization_22_1[1 * 256 * 1 * 1];
extern const float W53[256 * 128 * 3 * 3];
extern const float batch_normalization_23_1[1 * 128 * 1 * 1];
extern const float W52[128 * 256 * 1 * 1];
extern const float batch_normalization_24_1[1 * 256 * 1 * 1];
extern const float W51[256 * 128 * 3 * 3];
extern const float batch_normalization_25_1[1 * 128 * 1 * 1];
extern const float W50[128 * 256 * 1 * 1];
extern const float batch_normalization_26_1[1 * 256 * 1 * 1];
extern const float W49[256 * 128 * 3 * 3];
extern const float batch_normalization_27_1[1 * 512 * 1 * 1];
extern const float W48[512 * 256 * 3 * 3];
extern const float batch_normalization_28_1[1 * 256 * 1 * 1];
extern const float W47[256 * 512 * 1 * 1];
extern const float batch_normalization_29_1[1 * 512 * 1 * 1];
extern const float W46[512 * 256 * 3 * 3];
extern const float batch_normalization_30_1[1 * 256 * 1 * 1];
extern const float W45[256 * 512 * 1 * 1];
extern const float batch_normalization_31_1[1 * 512 * 1 * 1];
extern const float W44[512 * 256 * 3 * 3];
extern const float batch_normalization_32_1[1 * 256 * 1 * 1];
extern const float W43[256 * 512 * 1 * 1];
extern const float batch_normalization_33_1[1 * 512 * 1 * 1];
extern const float W42[512 * 256 * 3 * 3];
extern const float batch_normalization_34_1[1 * 256 * 1 * 1];
extern const float W41[256 * 512 * 1 * 1];
extern const float batch_normalization_35_1[1 * 512 * 1 * 1];
extern const float W40[512 * 256 * 3 * 3];
extern const float batch_normalization_36_1[1 * 256 * 1 * 1];
extern const float W39[256 * 512 * 1 * 1];
extern const float batch_normalization_37_1[1 * 512 * 1 * 1];
extern const float W38[512 * 256 * 3 * 3];
extern const float batch_normalization_38_1[1 * 256 * 1 * 1];
extern const float W37[256 * 512 * 1 * 1];
extern const float batch_normalization_39_1[1 * 512 * 1 * 1];
extern const float W36[512 * 256 * 3 * 3];
extern const float batch_normalization_40_1[1 * 256 * 1 * 1];
extern const float W35[256 * 512 * 1 * 1];
extern const float batch_normalization_41_1[1 * 512 * 1 * 1];
extern const float W34[512 * 256 * 3 * 3];
extern const float batch_normalization_42_1[1 * 256 * 1 * 1];
extern const float W33[256 * 512 * 1 * 1];
extern const float batch_normalization_43_1[1 * 512 * 1 * 1];
extern const float W32[512 * 256 * 3 * 3];
extern const float batch_normalization_44_1[1 * 1024 * 1 * 1];
extern const float W31[1024 * 512 * 3 * 3];
extern const float batch_normalization_45_1[1 * 512 * 1 * 1];
extern const float W30[512 * 1024 * 1 * 1];
extern const float batch_normalization_46_1[1 * 1024 * 1 * 1];
extern const float W29[1024 * 512 * 3 * 3];
extern const float batch_normalization_47_1[1 * 512 * 1 * 1];
extern const float W28[512 * 1024 * 1 * 1];
extern const float batch_normalization_48_1[1 * 1024 * 1 * 1];
extern const float W27[1024 * 512 * 3 * 3];
extern const float batch_normalization_49_1[1 * 512 * 1 * 1];
extern const float W26[512 * 1024 * 1 * 1];
extern const float batch_normalization_50_1[1 * 1024 * 1 * 1];
extern const float W25[1024 * 512 * 3 * 3];
extern const float batch_normalization_51_1[1 * 512 * 1 * 1];
extern const float W24[512 * 1024 * 1 * 1];
extern const float batch_normalization_52_1[1 * 1024 * 1 * 1];
extern const float W23[1024 * 512 * 3 * 3];
extern const float batch_normalization_53_1[1 * 512 * 1 * 1];
extern const float W22[512 * 1024 * 1 * 1];
extern const float batch_normalization_54_1[1 * 1024 * 1 * 1];
extern const float W21[1024 * 512 * 3 * 3];
extern const float batch_normalization_55_1[1 * 512 * 1 * 1];
extern const float W20[512 * 1024 * 1 * 1];
extern const float batch_normalization_56_1[1 * 1024 * 1 * 1];
extern const float W19[1024 * 512 * 3 * 3];
extern const float batch_normalization_57_1[1 * 512 * 1 * 1];
extern const float W18[512 * 1024 * 1 * 1];
extern const float batch_normalization_59_1[1 * 256 * 1 * 1];
extern const float W17[256 * 512 * 1 * 1];
extern const float batch_normalization_58_1[1 * 1024 * 1 * 1];
extern const float W5[1024 * 512 * 3 * 3];
extern const float batch_normalization_60_1[1 * 256 * 1 * 1];
extern const float W16[256 * 768 * 1 * 1];
extern const float batch_normalization_61_1[1 * 512 * 1 * 1];
extern const float W15[512 * 256 * 3 * 3];
extern const float batch_normalization_62_1[1 * 256 * 1 * 1];
extern const float W14[256 * 512 * 1 * 1];
extern const float batch_normalization_63_1[1 * 512 * 1 * 1];
extern const float W13[512 * 256 * 3 * 3];
extern const float batch_normalization_64_1[1 * 256 * 1 * 1];
extern const float W12[256 * 512 * 1 * 1];
extern const float batch_normalization_66_1[1 * 128 * 1 * 1];
extern const float W11[128 * 256 * 1 * 1];
extern const float batch_normalization_65_1[1 * 512 * 1 * 1];
extern const float W4[512 * 256 * 3 * 3];
extern const float batch_normalization_67_1[1 * 128 * 1 * 1];
extern const float W10[128 * 384 * 1 * 1];
extern const float batch_normalization_68_1[1 * 256 * 1 * 1];
extern const float W9[256 * 128 * 3 * 3];
extern const float batch_normalization_69_1[1 * 128 * 1 * 1];
extern const float W8[128 * 256 * 1 * 1];
extern const float batch_normalization_70_1[1 * 256 * 1 * 1];
extern const float W7[256 * 128 * 3 * 3];
extern const float batch_normalization_71_1[1 * 128 * 1 * 1];
extern const float W6[128 * 256 * 1 * 1];
extern const float batch_normalization_72_1[1 * 256 * 1 * 1];
extern const float W3[256 * 128 * 3 * 3];
extern "C" {
int yolo_v3(
    const float input_1[1 * 3 * 416 * 416], const unsigned src_w,
    const unsigned src_h,
    float selected_info[80 * 3 * ((13 * 13 + 26 * 26 + 52 * 52) / 169) * 5],
    unsigned selected_num[80]);
int yolo_v3_init();
int yolo_v3_fini();
int yolo_v3_helper(odla_computation comp);
};
static odla_computation Comp;
int yolo_v3_helper(odla_computation comp) {
  bool use_ipu_model = 0;
  int ipu_num = 1;
  int batches_per_step = 1;
  odla_SetComputationItem(comp, ODLA_USE_SIM_MODE,
                          (odla_item_value)&use_ipu_model);
  odla_SetComputationItem(comp, ODLA_PROCESSOR_NUM, (odla_item_value)&ipu_num);
  odla_SetComputationItem(comp, ODLA_BATCHES_PER_STEP,
                          (odla_item_value)&batches_per_step);
  auto input_1 =
      odla_CreateArgument({ODLA_FLOAT32, {.size = 4, .dims = {1, 3, 416, 416}}},
                          (const odla_value_id)("input_1"));
  auto src_h = odla_CreateArgument({ODLA_UINT32, {.size = 0, .dims = {}}},
                                   (const odla_value_id)("src_h"));
  auto src_w = odla_CreateArgument({ODLA_UINT32, {.size = 0, .dims = {}}},
                                   (const odla_value_id)("src_w"));
  auto W2_ = odla_CreateConstant(
      {ODLA_FLOAT32, {.size = 4, .dims = {255, 1024, 1, 1}}}, W2,
      (const odla_value_id) "W2_");
  auto W1_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {255, 512, 1, 1}}},
                          W1, (const odla_value_id) "W1_");
  auto W_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {255, 256, 1, 1}}},
                          W, (const odla_value_id) "W_");
  auto Resize_scales_ = odla_CreateConstant(
      {ODLA_FLOAT32, {.size = 1, .dims = {4}}}, Resize_scales,
      (const odla_value_id) "Resize_scales_");
  auto B2_ = odla_CreateConstant({ODLA_FLOAT32, {.size = 1, .dims = {255}}}, B2,
                                 (const odla_value_id) "B2_");
  auto B1_ = odla_CreateConstant({ODLA_FLOAT32, {.size = 1, .dims = {255}}}, B1,
                                 (const odla_value_id) "B1_");
  auto B_ = odla_CreateConstant({ODLA_FLOAT32, {.size = 1, .dims = {255}}}, B,
                                (const odla_value_id) "B_");
  auto batch_normalization_1_1_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {1, 32, 1, 1}}},
                          batch_normalization_1_1,
                          (const odla_value_id) "batch_normalization_1_1_");
  auto W74_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {32, 3, 3, 3}}},
                          W74, (const odla_value_id) "W74_");
  auto batch_normalization_2_1_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {1, 64, 1, 1}}},
                          batch_normalization_2_1,
                          (const odla_value_id) "batch_normalization_2_1_");
  auto W73_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {64, 32, 3, 3}}},
                          W73, (const odla_value_id) "W73_");
  auto batch_normalization_3_1_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {1, 32, 1, 1}}},
                          batch_normalization_3_1,
                          (const odla_value_id) "batch_normalization_3_1_");
  auto W72_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {32, 64, 1, 1}}},
                          W72, (const odla_value_id) "W72_");
  auto batch_normalization_4_1_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {1, 64, 1, 1}}},
                          batch_normalization_4_1,
                          (const odla_value_id) "batch_normalization_4_1_");
  auto W71_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {64, 32, 3, 3}}},
                          W71, (const odla_value_id) "W71_");
  auto batch_normalization_5_1_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {1, 128, 1, 1}}},
                          batch_normalization_5_1,
                          (const odla_value_id) "batch_normalization_5_1_");
  auto W70_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {128, 64, 3, 3}}},
                          W70, (const odla_value_id) "W70_");
  auto batch_normalization_6_1_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {1, 64, 1, 1}}},
                          batch_normalization_6_1,
                          (const odla_value_id) "batch_normalization_6_1_");
  auto W69_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {64, 128, 1, 1}}},
                          W69, (const odla_value_id) "W69_");
  auto batch_normalization_7_1_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {1, 128, 1, 1}}},
                          batch_normalization_7_1,
                          (const odla_value_id) "batch_normalization_7_1_");
  auto W68_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {128, 64, 3, 3}}},
                          W68, (const odla_value_id) "W68_");
  auto batch_normalization_8_1_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {1, 64, 1, 1}}},
                          batch_normalization_8_1,
                          (const odla_value_id) "batch_normalization_8_1_");
  auto W67_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {64, 128, 1, 1}}},
                          W67, (const odla_value_id) "W67_");
  auto batch_normalization_9_1_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {1, 128, 1, 1}}},
                          batch_normalization_9_1,
                          (const odla_value_id) "batch_normalization_9_1_");
  auto W66_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {128, 64, 3, 3}}},
                          W66, (const odla_value_id) "W66_");
  auto batch_normalization_10_1_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {1, 256, 1, 1}}},
                          batch_normalization_10_1,
                          (const odla_value_id) "batch_normalization_10_1_");
  auto W65_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {256, 128, 3, 3}}},
                          W65, (const odla_value_id) "W65_");
  auto batch_normalization_11_1_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {1, 128, 1, 1}}},
                          batch_normalization_11_1,
                          (const odla_value_id) "batch_normalization_11_1_");
  auto W64_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {128, 256, 1, 1}}},
                          W64, (const odla_value_id) "W64_");
  auto batch_normalization_12_1_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {1, 256, 1, 1}}},
                          batch_normalization_12_1,
                          (const odla_value_id) "batch_normalization_12_1_");
  auto W63_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {256, 128, 3, 3}}},
                          W63, (const odla_value_id) "W63_");
  auto batch_normalization_13_1_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {1, 128, 1, 1}}},
                          batch_normalization_13_1,
                          (const odla_value_id) "batch_normalization_13_1_");
  auto W62_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {128, 256, 1, 1}}},
                          W62, (const odla_value_id) "W62_");
  auto batch_normalization_14_1_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {1, 256, 1, 1}}},
                          batch_normalization_14_1,
                          (const odla_value_id) "batch_normalization_14_1_");
  auto W61_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {256, 128, 3, 3}}},
                          W61, (const odla_value_id) "W61_");
  auto batch_normalization_15_1_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {1, 128, 1, 1}}},
                          batch_normalization_15_1,
                          (const odla_value_id) "batch_normalization_15_1_");
  auto W60_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {128, 256, 1, 1}}},
                          W60, (const odla_value_id) "W60_");
  auto batch_normalization_16_1_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {1, 256, 1, 1}}},
                          batch_normalization_16_1,
                          (const odla_value_id) "batch_normalization_16_1_");
  auto W59_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {256, 128, 3, 3}}},
                          W59, (const odla_value_id) "W59_");
  auto batch_normalization_17_1_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {1, 128, 1, 1}}},
                          batch_normalization_17_1,
                          (const odla_value_id) "batch_normalization_17_1_");
  auto W58_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {128, 256, 1, 1}}},
                          W58, (const odla_value_id) "W58_");
  auto batch_normalization_18_1_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {1, 256, 1, 1}}},
                          batch_normalization_18_1,
                          (const odla_value_id) "batch_normalization_18_1_");
  auto W57_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {256, 128, 3, 3}}},
                          W57, (const odla_value_id) "W57_");
  auto batch_normalization_19_1_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {1, 128, 1, 1}}},
                          batch_normalization_19_1,
                          (const odla_value_id) "batch_normalization_19_1_");
  auto W56_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {128, 256, 1, 1}}},
                          W56, (const odla_value_id) "W56_");
  auto batch_normalization_20_1_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {1, 256, 1, 1}}},
                          batch_normalization_20_1,
                          (const odla_value_id) "batch_normalization_20_1_");
  auto W55_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {256, 128, 3, 3}}},
                          W55, (const odla_value_id) "W55_");
  auto batch_normalization_21_1_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {1, 128, 1, 1}}},
                          batch_normalization_21_1,
                          (const odla_value_id) "batch_normalization_21_1_");
  auto W54_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {128, 256, 1, 1}}},
                          W54, (const odla_value_id) "W54_");
  auto batch_normalization_22_1_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {1, 256, 1, 1}}},
                          batch_normalization_22_1,
                          (const odla_value_id) "batch_normalization_22_1_");
  auto W53_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {256, 128, 3, 3}}},
                          W53, (const odla_value_id) "W53_");
  auto batch_normalization_23_1_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {1, 128, 1, 1}}},
                          batch_normalization_23_1,
                          (const odla_value_id) "batch_normalization_23_1_");
  auto W52_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {128, 256, 1, 1}}},
                          W52, (const odla_value_id) "W52_");
  auto batch_normalization_24_1_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {1, 256, 1, 1}}},
                          batch_normalization_24_1,
                          (const odla_value_id) "batch_normalization_24_1_");
  auto W51_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {256, 128, 3, 3}}},
                          W51, (const odla_value_id) "W51_");
  auto batch_normalization_25_1_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {1, 128, 1, 1}}},
                          batch_normalization_25_1,
                          (const odla_value_id) "batch_normalization_25_1_");
  auto W50_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {128, 256, 1, 1}}},
                          W50, (const odla_value_id) "W50_");
  auto batch_normalization_26_1_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {1, 256, 1, 1}}},
                          batch_normalization_26_1,
                          (const odla_value_id) "batch_normalization_26_1_");
  auto W49_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {256, 128, 3, 3}}},
                          W49, (const odla_value_id) "W49_");
  auto batch_normalization_27_1_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {1, 512, 1, 1}}},
                          batch_normalization_27_1,
                          (const odla_value_id) "batch_normalization_27_1_");
  auto W48_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {512, 256, 3, 3}}},
                          W48, (const odla_value_id) "W48_");
  auto batch_normalization_28_1_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {1, 256, 1, 1}}},
                          batch_normalization_28_1,
                          (const odla_value_id) "batch_normalization_28_1_");
  auto W47_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {256, 512, 1, 1}}},
                          W47, (const odla_value_id) "W47_");
  auto batch_normalization_29_1_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {1, 512, 1, 1}}},
                          batch_normalization_29_1,
                          (const odla_value_id) "batch_normalization_29_1_");
  auto W46_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {512, 256, 3, 3}}},
                          W46, (const odla_value_id) "W46_");
  auto batch_normalization_30_1_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {1, 256, 1, 1}}},
                          batch_normalization_30_1,
                          (const odla_value_id) "batch_normalization_30_1_");
  auto W45_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {256, 512, 1, 1}}},
                          W45, (const odla_value_id) "W45_");
  auto batch_normalization_31_1_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {1, 512, 1, 1}}},
                          batch_normalization_31_1,
                          (const odla_value_id) "batch_normalization_31_1_");
  auto W44_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {512, 256, 3, 3}}},
                          W44, (const odla_value_id) "W44_");
  auto batch_normalization_32_1_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {1, 256, 1, 1}}},
                          batch_normalization_32_1,
                          (const odla_value_id) "batch_normalization_32_1_");
  auto W43_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {256, 512, 1, 1}}},
                          W43, (const odla_value_id) "W43_");
  auto batch_normalization_33_1_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {1, 512, 1, 1}}},
                          batch_normalization_33_1,
                          (const odla_value_id) "batch_normalization_33_1_");
  auto W42_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {512, 256, 3, 3}}},
                          W42, (const odla_value_id) "W42_");
  auto batch_normalization_34_1_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {1, 256, 1, 1}}},
                          batch_normalization_34_1,
                          (const odla_value_id) "batch_normalization_34_1_");
  auto W41_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {256, 512, 1, 1}}},
                          W41, (const odla_value_id) "W41_");
  auto batch_normalization_35_1_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {1, 512, 1, 1}}},
                          batch_normalization_35_1,
                          (const odla_value_id) "batch_normalization_35_1_");
  auto W40_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {512, 256, 3, 3}}},
                          W40, (const odla_value_id) "W40_");
  auto batch_normalization_36_1_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {1, 256, 1, 1}}},
                          batch_normalization_36_1,
                          (const odla_value_id) "batch_normalization_36_1_");
  auto W39_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {256, 512, 1, 1}}},
                          W39, (const odla_value_id) "W39_");
  auto batch_normalization_37_1_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {1, 512, 1, 1}}},
                          batch_normalization_37_1,
                          (const odla_value_id) "batch_normalization_37_1_");
  auto W38_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {512, 256, 3, 3}}},
                          W38, (const odla_value_id) "W38_");
  auto batch_normalization_38_1_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {1, 256, 1, 1}}},
                          batch_normalization_38_1,
                          (const odla_value_id) "batch_normalization_38_1_");
  auto W37_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {256, 512, 1, 1}}},
                          W37, (const odla_value_id) "W37_");
  auto batch_normalization_39_1_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {1, 512, 1, 1}}},
                          batch_normalization_39_1,
                          (const odla_value_id) "batch_normalization_39_1_");
  auto W36_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {512, 256, 3, 3}}},
                          W36, (const odla_value_id) "W36_");
  auto batch_normalization_40_1_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {1, 256, 1, 1}}},
                          batch_normalization_40_1,
                          (const odla_value_id) "batch_normalization_40_1_");
  auto W35_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {256, 512, 1, 1}}},
                          W35, (const odla_value_id) "W35_");
  auto batch_normalization_41_1_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {1, 512, 1, 1}}},
                          batch_normalization_41_1,
                          (const odla_value_id) "batch_normalization_41_1_");
  auto W34_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {512, 256, 3, 3}}},
                          W34, (const odla_value_id) "W34_");
  auto batch_normalization_42_1_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {1, 256, 1, 1}}},
                          batch_normalization_42_1,
                          (const odla_value_id) "batch_normalization_42_1_");
  auto W33_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {256, 512, 1, 1}}},
                          W33, (const odla_value_id) "W33_");
  auto batch_normalization_43_1_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {1, 512, 1, 1}}},
                          batch_normalization_43_1,
                          (const odla_value_id) "batch_normalization_43_1_");
  auto W32_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {512, 256, 3, 3}}},
                          W32, (const odla_value_id) "W32_");
  auto batch_normalization_44_1_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {1, 1024, 1, 1}}},
                          batch_normalization_44_1,
                          (const odla_value_id) "batch_normalization_44_1_");
  auto W31_ = odla_CreateConstant(
      {ODLA_FLOAT32, {.size = 4, .dims = {1024, 512, 3, 3}}}, W31,
      (const odla_value_id) "W31_");
  auto batch_normalization_45_1_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {1, 512, 1, 1}}},
                          batch_normalization_45_1,
                          (const odla_value_id) "batch_normalization_45_1_");
  auto W30_ = odla_CreateConstant(
      {ODLA_FLOAT32, {.size = 4, .dims = {512, 1024, 1, 1}}}, W30,
      (const odla_value_id) "W30_");
  auto batch_normalization_46_1_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {1, 1024, 1, 1}}},
                          batch_normalization_46_1,
                          (const odla_value_id) "batch_normalization_46_1_");
  auto W29_ = odla_CreateConstant(
      {ODLA_FLOAT32, {.size = 4, .dims = {1024, 512, 3, 3}}}, W29,
      (const odla_value_id) "W29_");
  auto batch_normalization_47_1_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {1, 512, 1, 1}}},
                          batch_normalization_47_1,
                          (const odla_value_id) "batch_normalization_47_1_");
  auto W28_ = odla_CreateConstant(
      {ODLA_FLOAT32, {.size = 4, .dims = {512, 1024, 1, 1}}}, W28,
      (const odla_value_id) "W28_");
  auto batch_normalization_48_1_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {1, 1024, 1, 1}}},
                          batch_normalization_48_1,
                          (const odla_value_id) "batch_normalization_48_1_");
  auto W27_ = odla_CreateConstant(
      {ODLA_FLOAT32, {.size = 4, .dims = {1024, 512, 3, 3}}}, W27,
      (const odla_value_id) "W27_");
  auto batch_normalization_49_1_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {1, 512, 1, 1}}},
                          batch_normalization_49_1,
                          (const odla_value_id) "batch_normalization_49_1_");
  auto W26_ = odla_CreateConstant(
      {ODLA_FLOAT32, {.size = 4, .dims = {512, 1024, 1, 1}}}, W26,
      (const odla_value_id) "W26_");
  auto batch_normalization_50_1_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {1, 1024, 1, 1}}},
                          batch_normalization_50_1,
                          (const odla_value_id) "batch_normalization_50_1_");
  auto W25_ = odla_CreateConstant(
      {ODLA_FLOAT32, {.size = 4, .dims = {1024, 512, 3, 3}}}, W25,
      (const odla_value_id) "W25_");
  auto batch_normalization_51_1_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {1, 512, 1, 1}}},
                          batch_normalization_51_1,
                          (const odla_value_id) "batch_normalization_51_1_");
  auto W24_ = odla_CreateConstant(
      {ODLA_FLOAT32, {.size = 4, .dims = {512, 1024, 1, 1}}}, W24,
      (const odla_value_id) "W24_");
  auto batch_normalization_52_1_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {1, 1024, 1, 1}}},
                          batch_normalization_52_1,
                          (const odla_value_id) "batch_normalization_52_1_");
  auto W23_ = odla_CreateConstant(
      {ODLA_FLOAT32, {.size = 4, .dims = {1024, 512, 3, 3}}}, W23,
      (const odla_value_id) "W23_");
  auto batch_normalization_53_1_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {1, 512, 1, 1}}},
                          batch_normalization_53_1,
                          (const odla_value_id) "batch_normalization_53_1_");
  auto W22_ = odla_CreateConstant(
      {ODLA_FLOAT32, {.size = 4, .dims = {512, 1024, 1, 1}}}, W22,
      (const odla_value_id) "W22_");
  auto batch_normalization_54_1_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {1, 1024, 1, 1}}},
                          batch_normalization_54_1,
                          (const odla_value_id) "batch_normalization_54_1_");
  auto W21_ = odla_CreateConstant(
      {ODLA_FLOAT32, {.size = 4, .dims = {1024, 512, 3, 3}}}, W21,
      (const odla_value_id) "W21_");
  auto batch_normalization_55_1_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {1, 512, 1, 1}}},
                          batch_normalization_55_1,
                          (const odla_value_id) "batch_normalization_55_1_");
  auto W20_ = odla_CreateConstant(
      {ODLA_FLOAT32, {.size = 4, .dims = {512, 1024, 1, 1}}}, W20,
      (const odla_value_id) "W20_");
  auto batch_normalization_56_1_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {1, 1024, 1, 1}}},
                          batch_normalization_56_1,
                          (const odla_value_id) "batch_normalization_56_1_");
  auto W19_ = odla_CreateConstant(
      {ODLA_FLOAT32, {.size = 4, .dims = {1024, 512, 3, 3}}}, W19,
      (const odla_value_id) "W19_");
  auto batch_normalization_57_1_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {1, 512, 1, 1}}},
                          batch_normalization_57_1,
                          (const odla_value_id) "batch_normalization_57_1_");
  auto W18_ = odla_CreateConstant(
      {ODLA_FLOAT32, {.size = 4, .dims = {512, 1024, 1, 1}}}, W18,
      (const odla_value_id) "W18_");
  auto batch_normalization_59_1_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {1, 256, 1, 1}}},
                          batch_normalization_59_1,
                          (const odla_value_id) "batch_normalization_59_1_");
  auto W17_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {256, 512, 1, 1}}},
                          W17, (const odla_value_id) "W17_");
  auto batch_normalization_58_1_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {1, 1024, 1, 1}}},
                          batch_normalization_58_1,
                          (const odla_value_id) "batch_normalization_58_1_");
  auto W5_ = odla_CreateConstant(
      {ODLA_FLOAT32, {.size = 4, .dims = {1024, 512, 3, 3}}}, W5,
      (const odla_value_id) "W5_");
  auto batch_normalization_60_1_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {1, 256, 1, 1}}},
                          batch_normalization_60_1,
                          (const odla_value_id) "batch_normalization_60_1_");
  auto W16_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {256, 768, 1, 1}}},
                          W16, (const odla_value_id) "W16_");
  auto batch_normalization_61_1_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {1, 512, 1, 1}}},
                          batch_normalization_61_1,
                          (const odla_value_id) "batch_normalization_61_1_");
  auto W15_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {512, 256, 3, 3}}},
                          W15, (const odla_value_id) "W15_");
  auto batch_normalization_62_1_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {1, 256, 1, 1}}},
                          batch_normalization_62_1,
                          (const odla_value_id) "batch_normalization_62_1_");
  auto W14_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {256, 512, 1, 1}}},
                          W14, (const odla_value_id) "W14_");
  auto batch_normalization_63_1_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {1, 512, 1, 1}}},
                          batch_normalization_63_1,
                          (const odla_value_id) "batch_normalization_63_1_");
  auto W13_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {512, 256, 3, 3}}},
                          W13, (const odla_value_id) "W13_");
  auto batch_normalization_64_1_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {1, 256, 1, 1}}},
                          batch_normalization_64_1,
                          (const odla_value_id) "batch_normalization_64_1_");
  auto W12_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {256, 512, 1, 1}}},
                          W12, (const odla_value_id) "W12_");
  auto batch_normalization_66_1_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {1, 128, 1, 1}}},
                          batch_normalization_66_1,
                          (const odla_value_id) "batch_normalization_66_1_");
  auto W11_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {128, 256, 1, 1}}},
                          W11, (const odla_value_id) "W11_");
  auto batch_normalization_65_1_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {1, 512, 1, 1}}},
                          batch_normalization_65_1,
                          (const odla_value_id) "batch_normalization_65_1_");
  auto W4_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {512, 256, 3, 3}}},
                          W4, (const odla_value_id) "W4_");
  auto batch_normalization_67_1_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {1, 128, 1, 1}}},
                          batch_normalization_67_1,
                          (const odla_value_id) "batch_normalization_67_1_");
  auto W10_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {128, 384, 1, 1}}},
                          W10, (const odla_value_id) "W10_");
  auto batch_normalization_68_1_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {1, 256, 1, 1}}},
                          batch_normalization_68_1,
                          (const odla_value_id) "batch_normalization_68_1_");
  auto W9_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {256, 128, 3, 3}}},
                          W9, (const odla_value_id) "W9_");
  auto batch_normalization_69_1_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {1, 128, 1, 1}}},
                          batch_normalization_69_1,
                          (const odla_value_id) "batch_normalization_69_1_");
  auto W8_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {128, 256, 1, 1}}},
                          W8, (const odla_value_id) "W8_");
  auto batch_normalization_70_1_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {1, 256, 1, 1}}},
                          batch_normalization_70_1,
                          (const odla_value_id) "batch_normalization_70_1_");
  auto W7_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {256, 128, 3, 3}}},
                          W7, (const odla_value_id) "W7_");
  auto batch_normalization_71_1_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {1, 128, 1, 1}}},
                          batch_normalization_71_1,
                          (const odla_value_id) "batch_normalization_71_1_");
  auto W6_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {128, 256, 1, 1}}},
                          W6, (const odla_value_id) "W6_");
  auto batch_normalization_72_1_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {1, 256, 1, 1}}},
                          batch_normalization_72_1,
                          (const odla_value_id) "batch_normalization_72_1_");
  auto W3_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 4, .dims = {256, 128, 3, 3}}},
                          W3, (const odla_value_id) "W3_");
  auto conv2d_1 = odla_Conv(
      input_1, odla_memory_layout::ODLA_CHANNELS_FIRST, 1, W74_,
      odla_memory_layout::ODLA_OIS, (const odla_uint32[]){1, 1},
      (const odla_uint32[]){1, 1}, (const odla_uint32[]){1, 1},
      (const odla_uint32[]){1, 1}, nullptr,
      {.size = 4, .dims = {1, 32, 416, 416}}, (const odla_value_id) "conv2d_1");
  auto batch_normalization_1_add =
      odla_Add(conv2d_1, batch_normalization_1_1_,
               (const odla_value_id) "batch_normalization_1_add");
  auto leaky_re_lu_1 = odla_LeakyRelu(batch_normalization_1_add, 0.1,
                                      (const odla_value_id) "leaky_re_lu_1");
  auto conv2d_2_0 =
      odla_Conv(leaky_re_lu_1, odla_memory_layout::ODLA_CHANNELS_FIRST, 1, W73_,
                odla_memory_layout::ODLA_OIS, (const odla_uint32[]){2, 2},
                (const odla_uint32[]){1, 1}, (const odla_uint32[]){1, 1},
                (const odla_uint32[]){0, 0}, nullptr,
                {.size = 4, .dims = {1, 64, 208, 208}},
                (const odla_value_id) "conv2d_2_0");
  auto batch_normalization_2_add =
      odla_Add(conv2d_2_0, batch_normalization_2_1_,
               (const odla_value_id) "batch_normalization_2_add");
  auto leaky_re_lu_2 = odla_LeakyRelu(batch_normalization_2_add, 0.1,
                                      (const odla_value_id) "leaky_re_lu_2");
  auto conv2d_3 = odla_Conv(
      leaky_re_lu_2, odla_memory_layout::ODLA_CHANNELS_FIRST, 1, W72_,
      odla_memory_layout::ODLA_OIS, (const odla_uint32[]){1, 1},
      (const odla_uint32[]){1, 1}, (const odla_uint32[]){0, 0},
      (const odla_uint32[]){0, 0}, nullptr,
      {.size = 4, .dims = {1, 32, 208, 208}}, (const odla_value_id) "conv2d_3");
  auto batch_normalization_3_add =
      odla_Add(conv2d_3, batch_normalization_3_1_,
               (const odla_value_id) "batch_normalization_3_add");
  auto leaky_re_lu_3 = odla_LeakyRelu(batch_normalization_3_add, 0.1,
                                      (const odla_value_id) "leaky_re_lu_3");
  auto conv2d_4 = odla_Conv(
      leaky_re_lu_3, odla_memory_layout::ODLA_CHANNELS_FIRST, 1, W71_,
      odla_memory_layout::ODLA_OIS, (const odla_uint32[]){1, 1},
      (const odla_uint32[]){1, 1}, (const odla_uint32[]){1, 1},
      (const odla_uint32[]){1, 1}, nullptr,
      {.size = 4, .dims = {1, 64, 208, 208}}, (const odla_value_id) "conv2d_4");
  auto batch_normalization_4_add =
      odla_Add(conv2d_4, batch_normalization_4_1_,
               (const odla_value_id) "batch_normalization_4_add");
  auto leaky_re_lu_4 = odla_LeakyRelu(batch_normalization_4_add, 0.1,
                                      (const odla_value_id) "leaky_re_lu_4");
  auto Add22 =
      odla_Add(leaky_re_lu_2, leaky_re_lu_4, (const odla_value_id) "Add22");
  auto conv2d_5_0 =
      odla_Conv(Add22, odla_memory_layout::ODLA_CHANNELS_FIRST, 1, W70_,
                odla_memory_layout::ODLA_OIS, (const odla_uint32[]){2, 2},
                (const odla_uint32[]){1, 1}, (const odla_uint32[]){1, 1},
                (const odla_uint32[]){0, 0}, nullptr,
                {.size = 4, .dims = {1, 128, 104, 104}},
                (const odla_value_id) "conv2d_5_0");
  auto batch_normalization_5_add =
      odla_Add(conv2d_5_0, batch_normalization_5_1_,
               (const odla_value_id) "batch_normalization_5_add");
  auto leaky_re_lu_5 = odla_LeakyRelu(batch_normalization_5_add, 0.1,
                                      (const odla_value_id) "leaky_re_lu_5");
  auto conv2d_6 = odla_Conv(
      leaky_re_lu_5, odla_memory_layout::ODLA_CHANNELS_FIRST, 1, W69_,
      odla_memory_layout::ODLA_OIS, (const odla_uint32[]){1, 1},
      (const odla_uint32[]){1, 1}, (const odla_uint32[]){0, 0},
      (const odla_uint32[]){0, 0}, nullptr,
      {.size = 4, .dims = {1, 64, 104, 104}}, (const odla_value_id) "conv2d_6");
  auto batch_normalization_6_add =
      odla_Add(conv2d_6, batch_normalization_6_1_,
               (const odla_value_id) "batch_normalization_6_add");
  auto leaky_re_lu_6 = odla_LeakyRelu(batch_normalization_6_add, 0.1,
                                      (const odla_value_id) "leaky_re_lu_6");
  auto conv2d_7 =
      odla_Conv(leaky_re_lu_6, odla_memory_layout::ODLA_CHANNELS_FIRST, 1, W68_,
                odla_memory_layout::ODLA_OIS, (const odla_uint32[]){1, 1},
                (const odla_uint32[]){1, 1}, (const odla_uint32[]){1, 1},
                (const odla_uint32[]){1, 1}, nullptr,
                {.size = 4, .dims = {1, 128, 104, 104}},
                (const odla_value_id) "conv2d_7");
  auto batch_normalization_7_add =
      odla_Add(conv2d_7, batch_normalization_7_1_,
               (const odla_value_id) "batch_normalization_7_add");
  auto leaky_re_lu_7 = odla_LeakyRelu(batch_normalization_7_add, 0.1,
                                      (const odla_value_id) "leaky_re_lu_7");
  auto Add21 =
      odla_Add(leaky_re_lu_5, leaky_re_lu_7, (const odla_value_id) "Add21");
  auto conv2d_8 = odla_Conv(
      Add21, odla_memory_layout::ODLA_CHANNELS_FIRST, 1, W67_,
      odla_memory_layout::ODLA_OIS, (const odla_uint32[]){1, 1},
      (const odla_uint32[]){1, 1}, (const odla_uint32[]){0, 0},
      (const odla_uint32[]){0, 0}, nullptr,
      {.size = 4, .dims = {1, 64, 104, 104}}, (const odla_value_id) "conv2d_8");
  auto batch_normalization_8_add =
      odla_Add(conv2d_8, batch_normalization_8_1_,
               (const odla_value_id) "batch_normalization_8_add");
  auto leaky_re_lu_8 = odla_LeakyRelu(batch_normalization_8_add, 0.1,
                                      (const odla_value_id) "leaky_re_lu_8");
  auto conv2d_9 =
      odla_Conv(leaky_re_lu_8, odla_memory_layout::ODLA_CHANNELS_FIRST, 1, W66_,
                odla_memory_layout::ODLA_OIS, (const odla_uint32[]){1, 1},
                (const odla_uint32[]){1, 1}, (const odla_uint32[]){1, 1},
                (const odla_uint32[]){1, 1}, nullptr,
                {.size = 4, .dims = {1, 128, 104, 104}},
                (const odla_value_id) "conv2d_9");
  auto batch_normalization_9_add =
      odla_Add(conv2d_9, batch_normalization_9_1_,
               (const odla_value_id) "batch_normalization_9_add");
  auto leaky_re_lu_9 = odla_LeakyRelu(batch_normalization_9_add, 0.1,
                                      (const odla_value_id) "leaky_re_lu_9");
  auto Add20 = odla_Add(Add21, leaky_re_lu_9, (const odla_value_id) "Add20");
  auto conv2d_10_0 =
      odla_Conv(Add20, odla_memory_layout::ODLA_CHANNELS_FIRST, 1, W65_,
                odla_memory_layout::ODLA_OIS, (const odla_uint32[]){2, 2},
                (const odla_uint32[]){1, 1}, (const odla_uint32[]){1, 1},
                (const odla_uint32[]){0, 0}, nullptr,
                {.size = 4, .dims = {1, 256, 52, 52}},
                (const odla_value_id) "conv2d_10_0");
  auto batch_normalization_10_add =
      odla_Add(conv2d_10_0, batch_normalization_10_1_,
               (const odla_value_id) "batch_normalization_10_add");
  auto leaky_re_lu_10 = odla_LeakyRelu(batch_normalization_10_add, 0.1,
                                       (const odla_value_id) "leaky_re_lu_10");
  auto conv2d_11 = odla_Conv(
      leaky_re_lu_10, odla_memory_layout::ODLA_CHANNELS_FIRST, 1, W64_,
      odla_memory_layout::ODLA_OIS, (const odla_uint32[]){1, 1},
      (const odla_uint32[]){1, 1}, (const odla_uint32[]){0, 0},
      (const odla_uint32[]){0, 0}, nullptr,
      {.size = 4, .dims = {1, 128, 52, 52}}, (const odla_value_id) "conv2d_11");
  auto batch_normalization_11_add =
      odla_Add(conv2d_11, batch_normalization_11_1_,
               (const odla_value_id) "batch_normalization_11_add");
  auto leaky_re_lu_11 = odla_LeakyRelu(batch_normalization_11_add, 0.1,
                                       (const odla_value_id) "leaky_re_lu_11");
  auto conv2d_12 = odla_Conv(
      leaky_re_lu_11, odla_memory_layout::ODLA_CHANNELS_FIRST, 1, W63_,
      odla_memory_layout::ODLA_OIS, (const odla_uint32[]){1, 1},
      (const odla_uint32[]){1, 1}, (const odla_uint32[]){1, 1},
      (const odla_uint32[]){1, 1}, nullptr,
      {.size = 4, .dims = {1, 256, 52, 52}}, (const odla_value_id) "conv2d_12");
  auto batch_normalization_12_add =
      odla_Add(conv2d_12, batch_normalization_12_1_,
               (const odla_value_id) "batch_normalization_12_add");
  auto leaky_re_lu_12 = odla_LeakyRelu(batch_normalization_12_add, 0.1,
                                       (const odla_value_id) "leaky_re_lu_12");
  auto Add19 =
      odla_Add(leaky_re_lu_10, leaky_re_lu_12, (const odla_value_id) "Add19");
  auto conv2d_13 = odla_Conv(
      Add19, odla_memory_layout::ODLA_CHANNELS_FIRST, 1, W62_,
      odla_memory_layout::ODLA_OIS, (const odla_uint32[]){1, 1},
      (const odla_uint32[]){1, 1}, (const odla_uint32[]){0, 0},
      (const odla_uint32[]){0, 0}, nullptr,
      {.size = 4, .dims = {1, 128, 52, 52}}, (const odla_value_id) "conv2d_13");
  auto batch_normalization_13_add =
      odla_Add(conv2d_13, batch_normalization_13_1_,
               (const odla_value_id) "batch_normalization_13_add");
  auto leaky_re_lu_13 = odla_LeakyRelu(batch_normalization_13_add, 0.1,
                                       (const odla_value_id) "leaky_re_lu_13");
  auto conv2d_14 = odla_Conv(
      leaky_re_lu_13, odla_memory_layout::ODLA_CHANNELS_FIRST, 1, W61_,
      odla_memory_layout::ODLA_OIS, (const odla_uint32[]){1, 1},
      (const odla_uint32[]){1, 1}, (const odla_uint32[]){1, 1},
      (const odla_uint32[]){1, 1}, nullptr,
      {.size = 4, .dims = {1, 256, 52, 52}}, (const odla_value_id) "conv2d_14");
  auto batch_normalization_14_add =
      odla_Add(conv2d_14, batch_normalization_14_1_,
               (const odla_value_id) "batch_normalization_14_add");
  auto leaky_re_lu_14 = odla_LeakyRelu(batch_normalization_14_add, 0.1,
                                       (const odla_value_id) "leaky_re_lu_14");
  auto Add18 = odla_Add(Add19, leaky_re_lu_14, (const odla_value_id) "Add18");
  auto conv2d_15 = odla_Conv(
      Add18, odla_memory_layout::ODLA_CHANNELS_FIRST, 1, W60_,
      odla_memory_layout::ODLA_OIS, (const odla_uint32[]){1, 1},
      (const odla_uint32[]){1, 1}, (const odla_uint32[]){0, 0},
      (const odla_uint32[]){0, 0}, nullptr,
      {.size = 4, .dims = {1, 128, 52, 52}}, (const odla_value_id) "conv2d_15");
  auto batch_normalization_15_add =
      odla_Add(conv2d_15, batch_normalization_15_1_,
               (const odla_value_id) "batch_normalization_15_add");
  auto leaky_re_lu_15 = odla_LeakyRelu(batch_normalization_15_add, 0.1,
                                       (const odla_value_id) "leaky_re_lu_15");
  auto conv2d_16 = odla_Conv(
      leaky_re_lu_15, odla_memory_layout::ODLA_CHANNELS_FIRST, 1, W59_,
      odla_memory_layout::ODLA_OIS, (const odla_uint32[]){1, 1},
      (const odla_uint32[]){1, 1}, (const odla_uint32[]){1, 1},
      (const odla_uint32[]){1, 1}, nullptr,
      {.size = 4, .dims = {1, 256, 52, 52}}, (const odla_value_id) "conv2d_16");
  auto batch_normalization_16_add =
      odla_Add(conv2d_16, batch_normalization_16_1_,
               (const odla_value_id) "batch_normalization_16_add");
  auto leaky_re_lu_16 = odla_LeakyRelu(batch_normalization_16_add, 0.1,
                                       (const odla_value_id) "leaky_re_lu_16");
  auto Add17 = odla_Add(Add18, leaky_re_lu_16, (const odla_value_id) "Add17");
  auto conv2d_17 = odla_Conv(
      Add17, odla_memory_layout::ODLA_CHANNELS_FIRST, 1, W58_,
      odla_memory_layout::ODLA_OIS, (const odla_uint32[]){1, 1},
      (const odla_uint32[]){1, 1}, (const odla_uint32[]){0, 0},
      (const odla_uint32[]){0, 0}, nullptr,
      {.size = 4, .dims = {1, 128, 52, 52}}, (const odla_value_id) "conv2d_17");
  auto batch_normalization_17_add =
      odla_Add(conv2d_17, batch_normalization_17_1_,
               (const odla_value_id) "batch_normalization_17_add");
  auto leaky_re_lu_17 = odla_LeakyRelu(batch_normalization_17_add, 0.1,
                                       (const odla_value_id) "leaky_re_lu_17");
  auto conv2d_18 = odla_Conv(
      leaky_re_lu_17, odla_memory_layout::ODLA_CHANNELS_FIRST, 1, W57_,
      odla_memory_layout::ODLA_OIS, (const odla_uint32[]){1, 1},
      (const odla_uint32[]){1, 1}, (const odla_uint32[]){1, 1},
      (const odla_uint32[]){1, 1}, nullptr,
      {.size = 4, .dims = {1, 256, 52, 52}}, (const odla_value_id) "conv2d_18");
  auto batch_normalization_18_add =
      odla_Add(conv2d_18, batch_normalization_18_1_,
               (const odla_value_id) "batch_normalization_18_add");
  auto leaky_re_lu_18 = odla_LeakyRelu(batch_normalization_18_add, 0.1,
                                       (const odla_value_id) "leaky_re_lu_18");
  auto Add16 = odla_Add(Add17, leaky_re_lu_18, (const odla_value_id) "Add16");
  auto conv2d_19 = odla_Conv(
      Add16, odla_memory_layout::ODLA_CHANNELS_FIRST, 1, W56_,
      odla_memory_layout::ODLA_OIS, (const odla_uint32[]){1, 1},
      (const odla_uint32[]){1, 1}, (const odla_uint32[]){0, 0},
      (const odla_uint32[]){0, 0}, nullptr,
      {.size = 4, .dims = {1, 128, 52, 52}}, (const odla_value_id) "conv2d_19");
  auto batch_normalization_19_add =
      odla_Add(conv2d_19, batch_normalization_19_1_,
               (const odla_value_id) "batch_normalization_19_add");
  auto leaky_re_lu_19 = odla_LeakyRelu(batch_normalization_19_add, 0.1,
                                       (const odla_value_id) "leaky_re_lu_19");
  auto conv2d_20 = odla_Conv(
      leaky_re_lu_19, odla_memory_layout::ODLA_CHANNELS_FIRST, 1, W55_,
      odla_memory_layout::ODLA_OIS, (const odla_uint32[]){1, 1},
      (const odla_uint32[]){1, 1}, (const odla_uint32[]){1, 1},
      (const odla_uint32[]){1, 1}, nullptr,
      {.size = 4, .dims = {1, 256, 52, 52}}, (const odla_value_id) "conv2d_20");
  auto batch_normalization_20_add =
      odla_Add(conv2d_20, batch_normalization_20_1_,
               (const odla_value_id) "batch_normalization_20_add");
  auto leaky_re_lu_20 = odla_LeakyRelu(batch_normalization_20_add, 0.1,
                                       (const odla_value_id) "leaky_re_lu_20");
  auto Add15 = odla_Add(Add16, leaky_re_lu_20, (const odla_value_id) "Add15");
  auto conv2d_21 = odla_Conv(
      Add15, odla_memory_layout::ODLA_CHANNELS_FIRST, 1, W54_,
      odla_memory_layout::ODLA_OIS, (const odla_uint32[]){1, 1},
      (const odla_uint32[]){1, 1}, (const odla_uint32[]){0, 0},
      (const odla_uint32[]){0, 0}, nullptr,
      {.size = 4, .dims = {1, 128, 52, 52}}, (const odla_value_id) "conv2d_21");
  auto batch_normalization_21_add =
      odla_Add(conv2d_21, batch_normalization_21_1_,
               (const odla_value_id) "batch_normalization_21_add");
  auto leaky_re_lu_21 = odla_LeakyRelu(batch_normalization_21_add, 0.1,
                                       (const odla_value_id) "leaky_re_lu_21");
  auto conv2d_22 = odla_Conv(
      leaky_re_lu_21, odla_memory_layout::ODLA_CHANNELS_FIRST, 1, W53_,
      odla_memory_layout::ODLA_OIS, (const odla_uint32[]){1, 1},
      (const odla_uint32[]){1, 1}, (const odla_uint32[]){1, 1},
      (const odla_uint32[]){1, 1}, nullptr,
      {.size = 4, .dims = {1, 256, 52, 52}}, (const odla_value_id) "conv2d_22");
  auto batch_normalization_22_add =
      odla_Add(conv2d_22, batch_normalization_22_1_,
               (const odla_value_id) "batch_normalization_22_add");
  auto leaky_re_lu_22 = odla_LeakyRelu(batch_normalization_22_add, 0.1,
                                       (const odla_value_id) "leaky_re_lu_22");
  auto Add14 = odla_Add(Add15, leaky_re_lu_22, (const odla_value_id) "Add14");
  auto conv2d_23 = odla_Conv(
      Add14, odla_memory_layout::ODLA_CHANNELS_FIRST, 1, W52_,
      odla_memory_layout::ODLA_OIS, (const odla_uint32[]){1, 1},
      (const odla_uint32[]){1, 1}, (const odla_uint32[]){0, 0},
      (const odla_uint32[]){0, 0}, nullptr,
      {.size = 4, .dims = {1, 128, 52, 52}}, (const odla_value_id) "conv2d_23");
  auto batch_normalization_23_add =
      odla_Add(conv2d_23, batch_normalization_23_1_,
               (const odla_value_id) "batch_normalization_23_add");
  auto leaky_re_lu_23 = odla_LeakyRelu(batch_normalization_23_add, 0.1,
                                       (const odla_value_id) "leaky_re_lu_23");
  auto conv2d_24 = odla_Conv(
      leaky_re_lu_23, odla_memory_layout::ODLA_CHANNELS_FIRST, 1, W51_,
      odla_memory_layout::ODLA_OIS, (const odla_uint32[]){1, 1},
      (const odla_uint32[]){1, 1}, (const odla_uint32[]){1, 1},
      (const odla_uint32[]){1, 1}, nullptr,
      {.size = 4, .dims = {1, 256, 52, 52}}, (const odla_value_id) "conv2d_24");
  auto batch_normalization_24_add =
      odla_Add(conv2d_24, batch_normalization_24_1_,
               (const odla_value_id) "batch_normalization_24_add");
  auto leaky_re_lu_24 = odla_LeakyRelu(batch_normalization_24_add, 0.1,
                                       (const odla_value_id) "leaky_re_lu_24");
  auto Add13 = odla_Add(Add14, leaky_re_lu_24, (const odla_value_id) "Add13");
  auto conv2d_25 = odla_Conv(
      Add13, odla_memory_layout::ODLA_CHANNELS_FIRST, 1, W50_,
      odla_memory_layout::ODLA_OIS, (const odla_uint32[]){1, 1},
      (const odla_uint32[]){1, 1}, (const odla_uint32[]){0, 0},
      (const odla_uint32[]){0, 0}, nullptr,
      {.size = 4, .dims = {1, 128, 52, 52}}, (const odla_value_id) "conv2d_25");
  auto batch_normalization_25_add =
      odla_Add(conv2d_25, batch_normalization_25_1_,
               (const odla_value_id) "batch_normalization_25_add");
  auto leaky_re_lu_25 = odla_LeakyRelu(batch_normalization_25_add, 0.1,
                                       (const odla_value_id) "leaky_re_lu_25");
  auto conv2d_26 = odla_Conv(
      leaky_re_lu_25, odla_memory_layout::ODLA_CHANNELS_FIRST, 1, W49_,
      odla_memory_layout::ODLA_OIS, (const odla_uint32[]){1, 1},
      (const odla_uint32[]){1, 1}, (const odla_uint32[]){1, 1},
      (const odla_uint32[]){1, 1}, nullptr,
      {.size = 4, .dims = {1, 256, 52, 52}}, (const odla_value_id) "conv2d_26");
  auto batch_normalization_26_add =
      odla_Add(conv2d_26, batch_normalization_26_1_,
               (const odla_value_id) "batch_normalization_26_add");
  auto leaky_re_lu_26 = odla_LeakyRelu(batch_normalization_26_add, 0.1,
                                       (const odla_value_id) "leaky_re_lu_26");
  auto Add = odla_Add(Add13, leaky_re_lu_26, (const odla_value_id) "Add");
  auto conv2d_27_0 =
      odla_Conv(Add, odla_memory_layout::ODLA_CHANNELS_FIRST, 1, W48_,
                odla_memory_layout::ODLA_OIS, (const odla_uint32[]){2, 2},
                (const odla_uint32[]){1, 1}, (const odla_uint32[]){1, 1},
                (const odla_uint32[]){0, 0}, nullptr,
                {.size = 4, .dims = {1, 512, 26, 26}},
                (const odla_value_id) "conv2d_27_0");
  auto batch_normalization_27_add =
      odla_Add(conv2d_27_0, batch_normalization_27_1_,
               (const odla_value_id) "batch_normalization_27_add");
  auto leaky_re_lu_27 = odla_LeakyRelu(batch_normalization_27_add, 0.1,
                                       (const odla_value_id) "leaky_re_lu_27");
  auto conv2d_28 = odla_Conv(
      leaky_re_lu_27, odla_memory_layout::ODLA_CHANNELS_FIRST, 1, W47_,
      odla_memory_layout::ODLA_OIS, (const odla_uint32[]){1, 1},
      (const odla_uint32[]){1, 1}, (const odla_uint32[]){0, 0},
      (const odla_uint32[]){0, 0}, nullptr,
      {.size = 4, .dims = {1, 256, 26, 26}}, (const odla_value_id) "conv2d_28");
  auto batch_normalization_28_add =
      odla_Add(conv2d_28, batch_normalization_28_1_,
               (const odla_value_id) "batch_normalization_28_add");
  auto leaky_re_lu_28 = odla_LeakyRelu(batch_normalization_28_add, 0.1,
                                       (const odla_value_id) "leaky_re_lu_28");
  auto conv2d_29 = odla_Conv(
      leaky_re_lu_28, odla_memory_layout::ODLA_CHANNELS_FIRST, 1, W46_,
      odla_memory_layout::ODLA_OIS, (const odla_uint32[]){1, 1},
      (const odla_uint32[]){1, 1}, (const odla_uint32[]){1, 1},
      (const odla_uint32[]){1, 1}, nullptr,
      {.size = 4, .dims = {1, 512, 26, 26}}, (const odla_value_id) "conv2d_29");
  auto batch_normalization_29_add =
      odla_Add(conv2d_29, batch_normalization_29_1_,
               (const odla_value_id) "batch_normalization_29_add");
  auto leaky_re_lu_29 = odla_LeakyRelu(batch_normalization_29_add, 0.1,
                                       (const odla_value_id) "leaky_re_lu_29");
  auto Add12 =
      odla_Add(leaky_re_lu_27, leaky_re_lu_29, (const odla_value_id) "Add12");
  auto conv2d_30 = odla_Conv(
      Add12, odla_memory_layout::ODLA_CHANNELS_FIRST, 1, W45_,
      odla_memory_layout::ODLA_OIS, (const odla_uint32[]){1, 1},
      (const odla_uint32[]){1, 1}, (const odla_uint32[]){0, 0},
      (const odla_uint32[]){0, 0}, nullptr,
      {.size = 4, .dims = {1, 256, 26, 26}}, (const odla_value_id) "conv2d_30");
  auto batch_normalization_30_add =
      odla_Add(conv2d_30, batch_normalization_30_1_,
               (const odla_value_id) "batch_normalization_30_add");
  auto leaky_re_lu_30 = odla_LeakyRelu(batch_normalization_30_add, 0.1,
                                       (const odla_value_id) "leaky_re_lu_30");
  auto conv2d_31 = odla_Conv(
      leaky_re_lu_30, odla_memory_layout::ODLA_CHANNELS_FIRST, 1, W44_,
      odla_memory_layout::ODLA_OIS, (const odla_uint32[]){1, 1},
      (const odla_uint32[]){1, 1}, (const odla_uint32[]){1, 1},
      (const odla_uint32[]){1, 1}, nullptr,
      {.size = 4, .dims = {1, 512, 26, 26}}, (const odla_value_id) "conv2d_31");
  auto batch_normalization_31_add =
      odla_Add(conv2d_31, batch_normalization_31_1_,
               (const odla_value_id) "batch_normalization_31_add");
  auto leaky_re_lu_31 = odla_LeakyRelu(batch_normalization_31_add, 0.1,
                                       (const odla_value_id) "leaky_re_lu_31");
  auto Add11 = odla_Add(Add12, leaky_re_lu_31, (const odla_value_id) "Add11");
  auto conv2d_32 = odla_Conv(
      Add11, odla_memory_layout::ODLA_CHANNELS_FIRST, 1, W43_,
      odla_memory_layout::ODLA_OIS, (const odla_uint32[]){1, 1},
      (const odla_uint32[]){1, 1}, (const odla_uint32[]){0, 0},
      (const odla_uint32[]){0, 0}, nullptr,
      {.size = 4, .dims = {1, 256, 26, 26}}, (const odla_value_id) "conv2d_32");
  auto batch_normalization_32_add =
      odla_Add(conv2d_32, batch_normalization_32_1_,
               (const odla_value_id) "batch_normalization_32_add");
  auto leaky_re_lu_32 = odla_LeakyRelu(batch_normalization_32_add, 0.1,
                                       (const odla_value_id) "leaky_re_lu_32");
  auto conv2d_33 = odla_Conv(
      leaky_re_lu_32, odla_memory_layout::ODLA_CHANNELS_FIRST, 1, W42_,
      odla_memory_layout::ODLA_OIS, (const odla_uint32[]){1, 1},
      (const odla_uint32[]){1, 1}, (const odla_uint32[]){1, 1},
      (const odla_uint32[]){1, 1}, nullptr,
      {.size = 4, .dims = {1, 512, 26, 26}}, (const odla_value_id) "conv2d_33");
  auto batch_normalization_33_add =
      odla_Add(conv2d_33, batch_normalization_33_1_,
               (const odla_value_id) "batch_normalization_33_add");
  auto leaky_re_lu_33 = odla_LeakyRelu(batch_normalization_33_add, 0.1,
                                       (const odla_value_id) "leaky_re_lu_33");
  auto Add10 = odla_Add(Add11, leaky_re_lu_33, (const odla_value_id) "Add10");
  auto conv2d_34 = odla_Conv(
      Add10, odla_memory_layout::ODLA_CHANNELS_FIRST, 1, W41_,
      odla_memory_layout::ODLA_OIS, (const odla_uint32[]){1, 1},
      (const odla_uint32[]){1, 1}, (const odla_uint32[]){0, 0},
      (const odla_uint32[]){0, 0}, nullptr,
      {.size = 4, .dims = {1, 256, 26, 26}}, (const odla_value_id) "conv2d_34");
  auto batch_normalization_34_add =
      odla_Add(conv2d_34, batch_normalization_34_1_,
               (const odla_value_id) "batch_normalization_34_add");
  auto leaky_re_lu_34 = odla_LeakyRelu(batch_normalization_34_add, 0.1,
                                       (const odla_value_id) "leaky_re_lu_34");
  auto conv2d_35 = odla_Conv(
      leaky_re_lu_34, odla_memory_layout::ODLA_CHANNELS_FIRST, 1, W40_,
      odla_memory_layout::ODLA_OIS, (const odla_uint32[]){1, 1},
      (const odla_uint32[]){1, 1}, (const odla_uint32[]){1, 1},
      (const odla_uint32[]){1, 1}, nullptr,
      {.size = 4, .dims = {1, 512, 26, 26}}, (const odla_value_id) "conv2d_35");
  auto batch_normalization_35_add =
      odla_Add(conv2d_35, batch_normalization_35_1_,
               (const odla_value_id) "batch_normalization_35_add");
  auto leaky_re_lu_35 = odla_LeakyRelu(batch_normalization_35_add, 0.1,
                                       (const odla_value_id) "leaky_re_lu_35");
  auto Add9 = odla_Add(Add10, leaky_re_lu_35, (const odla_value_id) "Add9");
  auto conv2d_36 = odla_Conv(
      Add9, odla_memory_layout::ODLA_CHANNELS_FIRST, 1, W39_,
      odla_memory_layout::ODLA_OIS, (const odla_uint32[]){1, 1},
      (const odla_uint32[]){1, 1}, (const odla_uint32[]){0, 0},
      (const odla_uint32[]){0, 0}, nullptr,
      {.size = 4, .dims = {1, 256, 26, 26}}, (const odla_value_id) "conv2d_36");
  auto batch_normalization_36_add =
      odla_Add(conv2d_36, batch_normalization_36_1_,
               (const odla_value_id) "batch_normalization_36_add");
  auto leaky_re_lu_36 = odla_LeakyRelu(batch_normalization_36_add, 0.1,
                                       (const odla_value_id) "leaky_re_lu_36");
  auto conv2d_37 = odla_Conv(
      leaky_re_lu_36, odla_memory_layout::ODLA_CHANNELS_FIRST, 1, W38_,
      odla_memory_layout::ODLA_OIS, (const odla_uint32[]){1, 1},
      (const odla_uint32[]){1, 1}, (const odla_uint32[]){1, 1},
      (const odla_uint32[]){1, 1}, nullptr,
      {.size = 4, .dims = {1, 512, 26, 26}}, (const odla_value_id) "conv2d_37");
  auto batch_normalization_37_add =
      odla_Add(conv2d_37, batch_normalization_37_1_,
               (const odla_value_id) "batch_normalization_37_add");
  auto leaky_re_lu_37 = odla_LeakyRelu(batch_normalization_37_add, 0.1,
                                       (const odla_value_id) "leaky_re_lu_37");
  auto Add8 = odla_Add(Add9, leaky_re_lu_37, (const odla_value_id) "Add8");
  auto conv2d_38 = odla_Conv(
      Add8, odla_memory_layout::ODLA_CHANNELS_FIRST, 1, W37_,
      odla_memory_layout::ODLA_OIS, (const odla_uint32[]){1, 1},
      (const odla_uint32[]){1, 1}, (const odla_uint32[]){0, 0},
      (const odla_uint32[]){0, 0}, nullptr,
      {.size = 4, .dims = {1, 256, 26, 26}}, (const odla_value_id) "conv2d_38");
  auto batch_normalization_38_add =
      odla_Add(conv2d_38, batch_normalization_38_1_,
               (const odla_value_id) "batch_normalization_38_add");
  auto leaky_re_lu_38 = odla_LeakyRelu(batch_normalization_38_add, 0.1,
                                       (const odla_value_id) "leaky_re_lu_38");
  auto conv2d_39 = odla_Conv(
      leaky_re_lu_38, odla_memory_layout::ODLA_CHANNELS_FIRST, 1, W36_,
      odla_memory_layout::ODLA_OIS, (const odla_uint32[]){1, 1},
      (const odla_uint32[]){1, 1}, (const odla_uint32[]){1, 1},
      (const odla_uint32[]){1, 1}, nullptr,
      {.size = 4, .dims = {1, 512, 26, 26}}, (const odla_value_id) "conv2d_39");
  auto batch_normalization_39_add =
      odla_Add(conv2d_39, batch_normalization_39_1_,
               (const odla_value_id) "batch_normalization_39_add");
  auto leaky_re_lu_39 = odla_LeakyRelu(batch_normalization_39_add, 0.1,
                                       (const odla_value_id) "leaky_re_lu_39");
  auto Add7 = odla_Add(Add8, leaky_re_lu_39, (const odla_value_id) "Add7");
  auto conv2d_40 = odla_Conv(
      Add7, odla_memory_layout::ODLA_CHANNELS_FIRST, 1, W35_,
      odla_memory_layout::ODLA_OIS, (const odla_uint32[]){1, 1},
      (const odla_uint32[]){1, 1}, (const odla_uint32[]){0, 0},
      (const odla_uint32[]){0, 0}, nullptr,
      {.size = 4, .dims = {1, 256, 26, 26}}, (const odla_value_id) "conv2d_40");
  auto batch_normalization_40_add =
      odla_Add(conv2d_40, batch_normalization_40_1_,
               (const odla_value_id) "batch_normalization_40_add");
  auto leaky_re_lu_40 = odla_LeakyRelu(batch_normalization_40_add, 0.1,
                                       (const odla_value_id) "leaky_re_lu_40");
  auto conv2d_41 = odla_Conv(
      leaky_re_lu_40, odla_memory_layout::ODLA_CHANNELS_FIRST, 1, W34_,
      odla_memory_layout::ODLA_OIS, (const odla_uint32[]){1, 1},
      (const odla_uint32[]){1, 1}, (const odla_uint32[]){1, 1},
      (const odla_uint32[]){1, 1}, nullptr,
      {.size = 4, .dims = {1, 512, 26, 26}}, (const odla_value_id) "conv2d_41");
  auto batch_normalization_41_add =
      odla_Add(conv2d_41, batch_normalization_41_1_,
               (const odla_value_id) "batch_normalization_41_add");
  auto leaky_re_lu_41 = odla_LeakyRelu(batch_normalization_41_add, 0.1,
                                       (const odla_value_id) "leaky_re_lu_41");
  auto Add6 = odla_Add(Add7, leaky_re_lu_41, (const odla_value_id) "Add6");
  auto conv2d_42 = odla_Conv(
      Add6, odla_memory_layout::ODLA_CHANNELS_FIRST, 1, W33_,
      odla_memory_layout::ODLA_OIS, (const odla_uint32[]){1, 1},
      (const odla_uint32[]){1, 1}, (const odla_uint32[]){0, 0},
      (const odla_uint32[]){0, 0}, nullptr,
      {.size = 4, .dims = {1, 256, 26, 26}}, (const odla_value_id) "conv2d_42");
  auto batch_normalization_42_add =
      odla_Add(conv2d_42, batch_normalization_42_1_,
               (const odla_value_id) "batch_normalization_42_add");
  auto leaky_re_lu_42 = odla_LeakyRelu(batch_normalization_42_add, 0.1,
                                       (const odla_value_id) "leaky_re_lu_42");
  auto conv2d_43 = odla_Conv(
      leaky_re_lu_42, odla_memory_layout::ODLA_CHANNELS_FIRST, 1, W32_,
      odla_memory_layout::ODLA_OIS, (const odla_uint32[]){1, 1},
      (const odla_uint32[]){1, 1}, (const odla_uint32[]){1, 1},
      (const odla_uint32[]){1, 1}, nullptr,
      {.size = 4, .dims = {1, 512, 26, 26}}, (const odla_value_id) "conv2d_43");
  auto batch_normalization_43_add =
      odla_Add(conv2d_43, batch_normalization_43_1_,
               (const odla_value_id) "batch_normalization_43_add");
  auto leaky_re_lu_43 = odla_LeakyRelu(batch_normalization_43_add, 0.1,
                                       (const odla_value_id) "leaky_re_lu_43");
  auto Add1 = odla_Add(Add6, leaky_re_lu_43, (const odla_value_id) "Add1");
  auto conv2d_44_0 =
      odla_Conv(Add1, odla_memory_layout::ODLA_CHANNELS_FIRST, 1, W31_,
                odla_memory_layout::ODLA_OIS, (const odla_uint32[]){2, 2},
                (const odla_uint32[]){1, 1}, (const odla_uint32[]){1, 1},
                (const odla_uint32[]){0, 0}, nullptr,
                {.size = 4, .dims = {1, 1024, 13, 13}},
                (const odla_value_id) "conv2d_44_0");
  auto batch_normalization_44_add =
      odla_Add(conv2d_44_0, batch_normalization_44_1_,
               (const odla_value_id) "batch_normalization_44_add");
  auto leaky_re_lu_44 = odla_LeakyRelu(batch_normalization_44_add, 0.1,
                                       (const odla_value_id) "leaky_re_lu_44");
  auto conv2d_45 = odla_Conv(
      leaky_re_lu_44, odla_memory_layout::ODLA_CHANNELS_FIRST, 1, W30_,
      odla_memory_layout::ODLA_OIS, (const odla_uint32[]){1, 1},
      (const odla_uint32[]){1, 1}, (const odla_uint32[]){0, 0},
      (const odla_uint32[]){0, 0}, nullptr,
      {.size = 4, .dims = {1, 512, 13, 13}}, (const odla_value_id) "conv2d_45");
  auto batch_normalization_45_add =
      odla_Add(conv2d_45, batch_normalization_45_1_,
               (const odla_value_id) "batch_normalization_45_add");
  auto leaky_re_lu_45 = odla_LeakyRelu(batch_normalization_45_add, 0.1,
                                       (const odla_value_id) "leaky_re_lu_45");
  auto conv2d_46 =
      odla_Conv(leaky_re_lu_45, odla_memory_layout::ODLA_CHANNELS_FIRST, 1,
                W29_, odla_memory_layout::ODLA_OIS, (const odla_uint32[]){1, 1},
                (const odla_uint32[]){1, 1}, (const odla_uint32[]){1, 1},
                (const odla_uint32[]){1, 1}, nullptr,
                {.size = 4, .dims = {1, 1024, 13, 13}},
                (const odla_value_id) "conv2d_46");
  auto batch_normalization_46_add =
      odla_Add(conv2d_46, batch_normalization_46_1_,
               (const odla_value_id) "batch_normalization_46_add");
  auto leaky_re_lu_46 = odla_LeakyRelu(batch_normalization_46_add, 0.1,
                                       (const odla_value_id) "leaky_re_lu_46");
  auto Add5 =
      odla_Add(leaky_re_lu_44, leaky_re_lu_46, (const odla_value_id) "Add5");
  auto conv2d_47 = odla_Conv(
      Add5, odla_memory_layout::ODLA_CHANNELS_FIRST, 1, W28_,
      odla_memory_layout::ODLA_OIS, (const odla_uint32[]){1, 1},
      (const odla_uint32[]){1, 1}, (const odla_uint32[]){0, 0},
      (const odla_uint32[]){0, 0}, nullptr,
      {.size = 4, .dims = {1, 512, 13, 13}}, (const odla_value_id) "conv2d_47");
  auto batch_normalization_47_add =
      odla_Add(conv2d_47, batch_normalization_47_1_,
               (const odla_value_id) "batch_normalization_47_add");
  auto leaky_re_lu_47 = odla_LeakyRelu(batch_normalization_47_add, 0.1,
                                       (const odla_value_id) "leaky_re_lu_47");
  auto conv2d_48 =
      odla_Conv(leaky_re_lu_47, odla_memory_layout::ODLA_CHANNELS_FIRST, 1,
                W27_, odla_memory_layout::ODLA_OIS, (const odla_uint32[]){1, 1},
                (const odla_uint32[]){1, 1}, (const odla_uint32[]){1, 1},
                (const odla_uint32[]){1, 1}, nullptr,
                {.size = 4, .dims = {1, 1024, 13, 13}},
                (const odla_value_id) "conv2d_48");
  auto batch_normalization_48_add =
      odla_Add(conv2d_48, batch_normalization_48_1_,
               (const odla_value_id) "batch_normalization_48_add");
  auto leaky_re_lu_48 = odla_LeakyRelu(batch_normalization_48_add, 0.1,
                                       (const odla_value_id) "leaky_re_lu_48");
  auto Add4 = odla_Add(Add5, leaky_re_lu_48, (const odla_value_id) "Add4");
  auto conv2d_49 = odla_Conv(
      Add4, odla_memory_layout::ODLA_CHANNELS_FIRST, 1, W26_,
      odla_memory_layout::ODLA_OIS, (const odla_uint32[]){1, 1},
      (const odla_uint32[]){1, 1}, (const odla_uint32[]){0, 0},
      (const odla_uint32[]){0, 0}, nullptr,
      {.size = 4, .dims = {1, 512, 13, 13}}, (const odla_value_id) "conv2d_49");
  auto batch_normalization_49_add =
      odla_Add(conv2d_49, batch_normalization_49_1_,
               (const odla_value_id) "batch_normalization_49_add");
  auto leaky_re_lu_49 = odla_LeakyRelu(batch_normalization_49_add, 0.1,
                                       (const odla_value_id) "leaky_re_lu_49");
  auto conv2d_50 =
      odla_Conv(leaky_re_lu_49, odla_memory_layout::ODLA_CHANNELS_FIRST, 1,
                W25_, odla_memory_layout::ODLA_OIS, (const odla_uint32[]){1, 1},
                (const odla_uint32[]){1, 1}, (const odla_uint32[]){1, 1},
                (const odla_uint32[]){1, 1}, nullptr,
                {.size = 4, .dims = {1, 1024, 13, 13}},
                (const odla_value_id) "conv2d_50");
  auto batch_normalization_50_add =
      odla_Add(conv2d_50, batch_normalization_50_1_,
               (const odla_value_id) "batch_normalization_50_add");
  auto leaky_re_lu_50 = odla_LeakyRelu(batch_normalization_50_add, 0.1,
                                       (const odla_value_id) "leaky_re_lu_50");
  auto Add3 = odla_Add(Add4, leaky_re_lu_50, (const odla_value_id) "Add3");
  auto conv2d_51 = odla_Conv(
      Add3, odla_memory_layout::ODLA_CHANNELS_FIRST, 1, W24_,
      odla_memory_layout::ODLA_OIS, (const odla_uint32[]){1, 1},
      (const odla_uint32[]){1, 1}, (const odla_uint32[]){0, 0},
      (const odla_uint32[]){0, 0}, nullptr,
      {.size = 4, .dims = {1, 512, 13, 13}}, (const odla_value_id) "conv2d_51");
  auto batch_normalization_51_add =
      odla_Add(conv2d_51, batch_normalization_51_1_,
               (const odla_value_id) "batch_normalization_51_add");
  auto leaky_re_lu_51 = odla_LeakyRelu(batch_normalization_51_add, 0.1,
                                       (const odla_value_id) "leaky_re_lu_51");
  auto conv2d_52 =
      odla_Conv(leaky_re_lu_51, odla_memory_layout::ODLA_CHANNELS_FIRST, 1,
                W23_, odla_memory_layout::ODLA_OIS, (const odla_uint32[]){1, 1},
                (const odla_uint32[]){1, 1}, (const odla_uint32[]){1, 1},
                (const odla_uint32[]){1, 1}, nullptr,
                {.size = 4, .dims = {1, 1024, 13, 13}},
                (const odla_value_id) "conv2d_52");
  auto batch_normalization_52_add =
      odla_Add(conv2d_52, batch_normalization_52_1_,
               (const odla_value_id) "batch_normalization_52_add");
  auto leaky_re_lu_52 = odla_LeakyRelu(batch_normalization_52_add, 0.1,
                                       (const odla_value_id) "leaky_re_lu_52");
  auto Add2 = odla_Add(Add3, leaky_re_lu_52, (const odla_value_id) "Add2");
  auto conv2d_53 = odla_Conv(
      Add2, odla_memory_layout::ODLA_CHANNELS_FIRST, 1, W22_,
      odla_memory_layout::ODLA_OIS, (const odla_uint32[]){1, 1},
      (const odla_uint32[]){1, 1}, (const odla_uint32[]){0, 0},
      (const odla_uint32[]){0, 0}, nullptr,
      {.size = 4, .dims = {1, 512, 13, 13}}, (const odla_value_id) "conv2d_53");
  auto batch_normalization_53_add =
      odla_Add(conv2d_53, batch_normalization_53_1_,
               (const odla_value_id) "batch_normalization_53_add");
  auto leaky_re_lu_53 = odla_LeakyRelu(batch_normalization_53_add, 0.1,
                                       (const odla_value_id) "leaky_re_lu_53");
  auto conv2d_54 =
      odla_Conv(leaky_re_lu_53, odla_memory_layout::ODLA_CHANNELS_FIRST, 1,
                W21_, odla_memory_layout::ODLA_OIS, (const odla_uint32[]){1, 1},
                (const odla_uint32[]){1, 1}, (const odla_uint32[]){1, 1},
                (const odla_uint32[]){1, 1}, nullptr,
                {.size = 4, .dims = {1, 1024, 13, 13}},
                (const odla_value_id) "conv2d_54");
  auto batch_normalization_54_add =
      odla_Add(conv2d_54, batch_normalization_54_1_,
               (const odla_value_id) "batch_normalization_54_add");
  auto leaky_re_lu_54 = odla_LeakyRelu(batch_normalization_54_add, 0.1,
                                       (const odla_value_id) "leaky_re_lu_54");
  auto conv2d_55 = odla_Conv(
      leaky_re_lu_54, odla_memory_layout::ODLA_CHANNELS_FIRST, 1, W20_,
      odla_memory_layout::ODLA_OIS, (const odla_uint32[]){1, 1},
      (const odla_uint32[]){1, 1}, (const odla_uint32[]){0, 0},
      (const odla_uint32[]){0, 0}, nullptr,
      {.size = 4, .dims = {1, 512, 13, 13}}, (const odla_value_id) "conv2d_55");
  auto batch_normalization_55_add =
      odla_Add(conv2d_55, batch_normalization_55_1_,
               (const odla_value_id) "batch_normalization_55_add");
  auto leaky_re_lu_55 = odla_LeakyRelu(batch_normalization_55_add, 0.1,
                                       (const odla_value_id) "leaky_re_lu_55");
  auto conv2d_56 =
      odla_Conv(leaky_re_lu_55, odla_memory_layout::ODLA_CHANNELS_FIRST, 1,
                W19_, odla_memory_layout::ODLA_OIS, (const odla_uint32[]){1, 1},
                (const odla_uint32[]){1, 1}, (const odla_uint32[]){1, 1},
                (const odla_uint32[]){1, 1}, nullptr,
                {.size = 4, .dims = {1, 1024, 13, 13}},
                (const odla_value_id) "conv2d_56");
  auto batch_normalization_56_add =
      odla_Add(conv2d_56, batch_normalization_56_1_,
               (const odla_value_id) "batch_normalization_56_add");
  auto leaky_re_lu_56 = odla_LeakyRelu(batch_normalization_56_add, 0.1,
                                       (const odla_value_id) "leaky_re_lu_56");
  auto conv2d_57 = odla_Conv(
      leaky_re_lu_56, odla_memory_layout::ODLA_CHANNELS_FIRST, 1, W18_,
      odla_memory_layout::ODLA_OIS, (const odla_uint32[]){1, 1},
      (const odla_uint32[]){1, 1}, (const odla_uint32[]){0, 0},
      (const odla_uint32[]){0, 0}, nullptr,
      {.size = 4, .dims = {1, 512, 13, 13}}, (const odla_value_id) "conv2d_57");
  auto batch_normalization_57_add =
      odla_Add(conv2d_57, batch_normalization_57_1_,
               (const odla_value_id) "batch_normalization_57_add");
  auto leaky_re_lu_57 = odla_LeakyRelu(batch_normalization_57_add, 0.1,
                                       (const odla_value_id) "leaky_re_lu_57");
  auto conv2d_60 = odla_Conv(
      leaky_re_lu_57, odla_memory_layout::ODLA_CHANNELS_FIRST, 1, W17_,
      odla_memory_layout::ODLA_OIS, (const odla_uint32[]){1, 1},
      (const odla_uint32[]){1, 1}, (const odla_uint32[]){0, 0},
      (const odla_uint32[]){0, 0}, nullptr,
      {.size = 4, .dims = {1, 256, 13, 13}}, (const odla_value_id) "conv2d_60");
  auto batch_normalization_59_add =
      odla_Add(conv2d_60, batch_normalization_59_1_,
               (const odla_value_id) "batch_normalization_59_add");
  auto leaky_re_lu_59 = odla_LeakyRelu(batch_normalization_59_add, 0.1,
                                       (const odla_value_id) "leaky_re_lu_59");
  auto Resize1 = odla_Resize(leaky_re_lu_59, ODLA_NEAREST, ODLA_HALF_PIXEL, -1,
                             {.size = 4, .dims = {1, 256, 26, 26}},
                             (const odla_value_id) "Resize1");
  auto conv2d_58 =
      odla_Conv(leaky_re_lu_57, odla_memory_layout::ODLA_CHANNELS_FIRST, 1, W5_,
                odla_memory_layout::ODLA_OIS, (const odla_uint32[]){1, 1},
                (const odla_uint32[]){1, 1}, (const odla_uint32[]){1, 1},
                (const odla_uint32[]){1, 1}, nullptr,
                {.size = 4, .dims = {1, 1024, 13, 13}},
                (const odla_value_id) "conv2d_58");
  auto batch_normalization_58_add =
      odla_Add(conv2d_58, batch_normalization_58_1_,
               (const odla_value_id) "batch_normalization_58_add");
  auto leaky_re_lu_58 = odla_LeakyRelu(batch_normalization_58_add, 0.1,
                                       (const odla_value_id) "leaky_re_lu_58");
  auto conv2d_59 = odla_Conv(
      leaky_re_lu_58, odla_memory_layout::ODLA_CHANNELS_FIRST, 1, W2_,
      odla_memory_layout::ODLA_OIS, (const odla_uint32[]){1, 1},
      (const odla_uint32[]){1, 1}, (const odla_uint32[]){0, 0},
      (const odla_uint32[]){0, 0}, B2_, {.size = 4, .dims = {1, 255, 13, 13}},
      (const odla_value_id) "conv2d_59");
  auto concatenate_1 = odla_Concat((odla_values){.size = 2,
                                                 .values =
                                                     {
                                                         Resize1,
                                                         Add1,
                                                     }},
                                   1, {.size = 4, .dims = {1, 768, 26, 26}},
                                   (const odla_value_id) "concatenate_1");
  auto conv2d_61 = odla_Conv(
      concatenate_1, odla_memory_layout::ODLA_CHANNELS_FIRST, 1, W16_,
      odla_memory_layout::ODLA_OIS, (const odla_uint32[]){1, 1},
      (const odla_uint32[]){1, 1}, (const odla_uint32[]){0, 0},
      (const odla_uint32[]){0, 0}, nullptr,
      {.size = 4, .dims = {1, 256, 26, 26}}, (const odla_value_id) "conv2d_61");
  auto batch_normalization_60_add =
      odla_Add(conv2d_61, batch_normalization_60_1_,
               (const odla_value_id) "batch_normalization_60_add");
  auto leaky_re_lu_60 = odla_LeakyRelu(batch_normalization_60_add, 0.1,
                                       (const odla_value_id) "leaky_re_lu_60");
  auto conv2d_62 = odla_Conv(
      leaky_re_lu_60, odla_memory_layout::ODLA_CHANNELS_FIRST, 1, W15_,
      odla_memory_layout::ODLA_OIS, (const odla_uint32[]){1, 1},
      (const odla_uint32[]){1, 1}, (const odla_uint32[]){1, 1},
      (const odla_uint32[]){1, 1}, nullptr,
      {.size = 4, .dims = {1, 512, 26, 26}}, (const odla_value_id) "conv2d_62");
  auto batch_normalization_61_add =
      odla_Add(conv2d_62, batch_normalization_61_1_,
               (const odla_value_id) "batch_normalization_61_add");
  auto leaky_re_lu_61 = odla_LeakyRelu(batch_normalization_61_add, 0.1,
                                       (const odla_value_id) "leaky_re_lu_61");
  auto conv2d_63 = odla_Conv(
      leaky_re_lu_61, odla_memory_layout::ODLA_CHANNELS_FIRST, 1, W14_,
      odla_memory_layout::ODLA_OIS, (const odla_uint32[]){1, 1},
      (const odla_uint32[]){1, 1}, (const odla_uint32[]){0, 0},
      (const odla_uint32[]){0, 0}, nullptr,
      {.size = 4, .dims = {1, 256, 26, 26}}, (const odla_value_id) "conv2d_63");
  auto batch_normalization_62_add =
      odla_Add(conv2d_63, batch_normalization_62_1_,
               (const odla_value_id) "batch_normalization_62_add");
  auto leaky_re_lu_62 = odla_LeakyRelu(batch_normalization_62_add, 0.1,
                                       (const odla_value_id) "leaky_re_lu_62");
  auto conv2d_64 = odla_Conv(
      leaky_re_lu_62, odla_memory_layout::ODLA_CHANNELS_FIRST, 1, W13_,
      odla_memory_layout::ODLA_OIS, (const odla_uint32[]){1, 1},
      (const odla_uint32[]){1, 1}, (const odla_uint32[]){1, 1},
      (const odla_uint32[]){1, 1}, nullptr,
      {.size = 4, .dims = {1, 512, 26, 26}}, (const odla_value_id) "conv2d_64");
  auto batch_normalization_63_add =
      odla_Add(conv2d_64, batch_normalization_63_1_,
               (const odla_value_id) "batch_normalization_63_add");
  auto leaky_re_lu_63 = odla_LeakyRelu(batch_normalization_63_add, 0.1,
                                       (const odla_value_id) "leaky_re_lu_63");
  auto conv2d_65 = odla_Conv(
      leaky_re_lu_63, odla_memory_layout::ODLA_CHANNELS_FIRST, 1, W12_,
      odla_memory_layout::ODLA_OIS, (const odla_uint32[]){1, 1},
      (const odla_uint32[]){1, 1}, (const odla_uint32[]){0, 0},
      (const odla_uint32[]){0, 0}, nullptr,
      {.size = 4, .dims = {1, 256, 26, 26}}, (const odla_value_id) "conv2d_65");
  auto batch_normalization_64_add =
      odla_Add(conv2d_65, batch_normalization_64_1_,
               (const odla_value_id) "batch_normalization_64_add");
  auto leaky_re_lu_64 = odla_LeakyRelu(batch_normalization_64_add, 0.1,
                                       (const odla_value_id) "leaky_re_lu_64");
  auto conv2d_68 = odla_Conv(
      leaky_re_lu_64, odla_memory_layout::ODLA_CHANNELS_FIRST, 1, W11_,
      odla_memory_layout::ODLA_OIS, (const odla_uint32[]){1, 1},
      (const odla_uint32[]){1, 1}, (const odla_uint32[]){0, 0},
      (const odla_uint32[]){0, 0}, nullptr,
      {.size = 4, .dims = {1, 128, 26, 26}}, (const odla_value_id) "conv2d_68");
  auto batch_normalization_66_add =
      odla_Add(conv2d_68, batch_normalization_66_1_,
               (const odla_value_id) "batch_normalization_66_add");
  auto leaky_re_lu_66 = odla_LeakyRelu(batch_normalization_66_add, 0.1,
                                       (const odla_value_id) "leaky_re_lu_66");
  auto Resize = odla_Resize(leaky_re_lu_66, ODLA_NEAREST, ODLA_HALF_PIXEL, -1,
                            {.size = 4, .dims = {1, 128, 52, 52}},
                            (const odla_value_id) "Resize");
  auto conv2d_66 = odla_Conv(
      leaky_re_lu_64, odla_memory_layout::ODLA_CHANNELS_FIRST, 1, W4_,
      odla_memory_layout::ODLA_OIS, (const odla_uint32[]){1, 1},
      (const odla_uint32[]){1, 1}, (const odla_uint32[]){1, 1},
      (const odla_uint32[]){1, 1}, nullptr,
      {.size = 4, .dims = {1, 512, 26, 26}}, (const odla_value_id) "conv2d_66");
  auto batch_normalization_65_add =
      odla_Add(conv2d_66, batch_normalization_65_1_,
               (const odla_value_id) "batch_normalization_65_add");
  auto leaky_re_lu_65 = odla_LeakyRelu(batch_normalization_65_add, 0.1,
                                       (const odla_value_id) "leaky_re_lu_65");
  auto conv2d_67 = odla_Conv(
      leaky_re_lu_65, odla_memory_layout::ODLA_CHANNELS_FIRST, 1, W1_,
      odla_memory_layout::ODLA_OIS, (const odla_uint32[]){1, 1},
      (const odla_uint32[]){1, 1}, (const odla_uint32[]){0, 0},
      (const odla_uint32[]){0, 0}, B1_, {.size = 4, .dims = {1, 255, 26, 26}},
      (const odla_value_id) "conv2d_67");
  auto concatenate_2 = odla_Concat((odla_values){.size = 2,
                                                 .values =
                                                     {
                                                         Resize,
                                                         Add,
                                                     }},
                                   1, {.size = 4, .dims = {1, 384, 52, 52}},
                                   (const odla_value_id) "concatenate_2");
  auto conv2d_69 = odla_Conv(
      concatenate_2, odla_memory_layout::ODLA_CHANNELS_FIRST, 1, W10_,
      odla_memory_layout::ODLA_OIS, (const odla_uint32[]){1, 1},
      (const odla_uint32[]){1, 1}, (const odla_uint32[]){0, 0},
      (const odla_uint32[]){0, 0}, nullptr,
      {.size = 4, .dims = {1, 128, 52, 52}}, (const odla_value_id) "conv2d_69");
  auto batch_normalization_67_add =
      odla_Add(conv2d_69, batch_normalization_67_1_,
               (const odla_value_id) "batch_normalization_67_add");
  auto leaky_re_lu_67 = odla_LeakyRelu(batch_normalization_67_add, 0.1,
                                       (const odla_value_id) "leaky_re_lu_67");
  auto conv2d_70 = odla_Conv(
      leaky_re_lu_67, odla_memory_layout::ODLA_CHANNELS_FIRST, 1, W9_,
      odla_memory_layout::ODLA_OIS, (const odla_uint32[]){1, 1},
      (const odla_uint32[]){1, 1}, (const odla_uint32[]){1, 1},
      (const odla_uint32[]){1, 1}, nullptr,
      {.size = 4, .dims = {1, 256, 52, 52}}, (const odla_value_id) "conv2d_70");
  auto batch_normalization_68_add =
      odla_Add(conv2d_70, batch_normalization_68_1_,
               (const odla_value_id) "batch_normalization_68_add");
  auto leaky_re_lu_68 = odla_LeakyRelu(batch_normalization_68_add, 0.1,
                                       (const odla_value_id) "leaky_re_lu_68");
  auto conv2d_71 = odla_Conv(
      leaky_re_lu_68, odla_memory_layout::ODLA_CHANNELS_FIRST, 1, W8_,
      odla_memory_layout::ODLA_OIS, (const odla_uint32[]){1, 1},
      (const odla_uint32[]){1, 1}, (const odla_uint32[]){0, 0},
      (const odla_uint32[]){0, 0}, nullptr,
      {.size = 4, .dims = {1, 128, 52, 52}}, (const odla_value_id) "conv2d_71");
  auto batch_normalization_69_add =
      odla_Add(conv2d_71, batch_normalization_69_1_,
               (const odla_value_id) "batch_normalization_69_add");
  auto leaky_re_lu_69 = odla_LeakyRelu(batch_normalization_69_add, 0.1,
                                       (const odla_value_id) "leaky_re_lu_69");
  auto conv2d_72 = odla_Conv(
      leaky_re_lu_69, odla_memory_layout::ODLA_CHANNELS_FIRST, 1, W7_,
      odla_memory_layout::ODLA_OIS, (const odla_uint32[]){1, 1},
      (const odla_uint32[]){1, 1}, (const odla_uint32[]){1, 1},
      (const odla_uint32[]){1, 1}, nullptr,
      {.size = 4, .dims = {1, 256, 52, 52}}, (const odla_value_id) "conv2d_72");
  auto batch_normalization_70_add =
      odla_Add(conv2d_72, batch_normalization_70_1_,
               (const odla_value_id) "batch_normalization_70_add");
  auto leaky_re_lu_70 = odla_LeakyRelu(batch_normalization_70_add, 0.1,
                                       (const odla_value_id) "leaky_re_lu_70");
  auto conv2d_73 = odla_Conv(
      leaky_re_lu_70, odla_memory_layout::ODLA_CHANNELS_FIRST, 1, W6_,
      odla_memory_layout::ODLA_OIS, (const odla_uint32[]){1, 1},
      (const odla_uint32[]){1, 1}, (const odla_uint32[]){0, 0},
      (const odla_uint32[]){0, 0}, nullptr,
      {.size = 4, .dims = {1, 128, 52, 52}}, (const odla_value_id) "conv2d_73");
  auto batch_normalization_71_add =
      odla_Add(conv2d_73, batch_normalization_71_1_,
               (const odla_value_id) "batch_normalization_71_add");
  auto leaky_re_lu_71 = odla_LeakyRelu(batch_normalization_71_add, 0.1,
                                       (const odla_value_id) "leaky_re_lu_71");
  auto conv2d_74 = odla_Conv(
      leaky_re_lu_71, odla_memory_layout::ODLA_CHANNELS_FIRST, 1, W3_,
      odla_memory_layout::ODLA_OIS, (const odla_uint32[]){1, 1},
      (const odla_uint32[]){1, 1}, (const odla_uint32[]){1, 1},
      (const odla_uint32[]){1, 1}, nullptr,
      {.size = 4, .dims = {1, 256, 52, 52}}, (const odla_value_id) "conv2d_74");
  auto batch_normalization_72_add =
      odla_Add(conv2d_74, batch_normalization_72_1_,
               (const odla_value_id) "batch_normalization_72_add");
  auto leaky_re_lu_72 = odla_LeakyRelu(batch_normalization_72_add, 0.1,
                                       (const odla_value_id) "leaky_re_lu_72");
  auto conv2d_75 = odla_Conv(
      leaky_re_lu_72, odla_memory_layout::ODLA_CHANNELS_FIRST, 1, W_,
      odla_memory_layout::ODLA_OIS, (const odla_uint32[]){1, 1},
      (const odla_uint32[]){1, 1}, (const odla_uint32[]){0, 0},
      (const odla_uint32[]){0, 0}, B_, {.size = 4, .dims = {1, 255, 52, 52}},
      (const odla_value_id) "conv2d_75");
  auto post_process_outs =
      odla_PostProcess(src_w, src_h, conv2d_59, conv2d_67, conv2d_75,
                       (const odla_value_id) "post_process_outs");
  odla_SetValuesAsOutput(post_process_outs);
  return ODLA_SUCCESS;
}
int yolo_v3_fini() {
  if (Comp != nullptr) {
    return odla_DestroyComputation(Comp);
  }
  return ODLA_SUCCESS;
}
int yolo_v3_init() {
  odla_status status = ODLA_SUCCESS;
  if (Comp == nullptr) {
    status = odla_CreateComputation(&Comp);
    if (status != ODLA_SUCCESS) {
      return status;
    }
    status = (odla_status)yolo_v3_helper(Comp);
  }
  return status;
}
int yolo_v3(
    const float input_1[1 * 3 * 416 * 416], const unsigned src_w,
    const unsigned src_h,
    float selected_info[80 * 3 * ((13 * 13 + 26 * 26 + 52 * 52) / 169) * 5],
    unsigned selected_num[80]) {
  odla_status status = ODLA_SUCCESS;
  status = (odla_status)yolo_v3_init();
  if (status != ODLA_SUCCESS) {
    return status;
  }
  static odla_context Ctx;
  if (Ctx == nullptr) {
    status = odla_CreateContext(&Ctx);
    if (status != ODLA_SUCCESS) {
      return status;
    }
  }
  status =
      odla_BindToArgumentById((const odla_value_id) "input_1", input_1, Ctx);
  if (status != ODLA_SUCCESS) {
    return status;
  }
  status = odla_BindToArgumentById((const odla_value_id) "src_w", &src_w, Ctx);
  if (status != ODLA_SUCCESS) {
    return status;
  }
  status = odla_BindToArgumentById((const odla_value_id) "src_h", &src_h, Ctx);
  if (status != ODLA_SUCCESS) {
    return status;
  }
  status = odla_BindToOutputById((const odla_value_id) "post_process_outs0",
                                 selected_info, Ctx);
  if (status != ODLA_SUCCESS) {
    return status;
  }
  status = odla_BindToOutputById((const odla_value_id) "post_process_outs1",
                                 selected_num, Ctx);
  if (status != ODLA_SUCCESS) {
    return status;
  }
  auto begin = clock();
  puts("ExecuteComputation starts.\n");
  auto exec =
      odla_ExecuteComputation(Comp, Ctx, ODLA_COMPUTE_INFERENCE, nullptr);
  auto end = clock();
  printf("ExecuteComputation ends: %lf seconds.\n",
         (end - begin) * 1.0 / CLOCKS_PER_SEC);
  return exec;
}
