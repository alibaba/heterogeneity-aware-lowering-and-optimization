//===- yolov3_test.cc ----------------------------------------------------===//
//
// Copyright (C) 2020-2021 Alibaba Group Holding Limited.
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

// clang-format off
// Testing CXX Code Gen using ODLA on TensorRT
// RUN: %halo_compiler --disable-broadcasting -target cxx %src_dir/tests/parser/onnx/yolov3-10.onnx -outputs conv2d_59 -outputs conv2d_67 -outputs conv2d_75 -input-shape=input_1:1x3x416x416 -entry-func-name=yolo_v3 -o %t.cc
// RUN: %cxx -c -o %t.o %t.cc -I%odla_path/include
// RUN: %cxx -c -o %t_pp.o %p/yolov3_postproc.cc
// RUN: %cxx -c -DUSE_NCHW_DATA -DCOMPARE_ERROR=1e-2 %include %s -o %t_main.o
// RUN: %cxx %t.o %t.bin %t_main.o %t_pp.o %odla_link -lodla_tensorrt -o %t.exe

// ODLA/XNNPACK
// RUN: %halo_compiler -exec-mode=interpret -emit-value-id-as-int --disable-broadcasting -target cc %src_dir/tests/parser/onnx/yolov3-10.onnx -outputs conv2d_59 -outputs conv2d_67 -outputs conv2d_75 -input-shape=input_1:1x3x416x416 -entry-func-name=yolo_v3 -remove-input-transpose -remove-output-transpose -reorder-data-layout=channel-last -o %t.c
// RUN: %cc -c -o %t.o %t.c -I%odla_path/include
// RUN: %cxx -c -DCOMPARE_ERROR=1e-3 %include %s -o %t_main.o
// RUN: %cxx %t.o %t.bin %t_main.o %t_pp.o %odla_link -lodla_xnnpack -o %t.exe
// RUN: %t.exe | FileCheck %s

// ODLA/QAIC
// RUN: %halo_compiler --disable-broadcasting -target cc %src_dir/tests/parser/onnx/yolov3-10.onnx -outputs conv2d_59 -outputs conv2d_67 -outputs conv2d_75 -input-shape=input_1:1x3x416x416 -entry-func-name=yolo_v3 -remove-input-transpose -remove-output-transpose -reorder-data-layout=channel-last -o %t_qaic.c
// RUN: %cc -c -o %t_qaic.o %t_qaic.c -I%odla_path/include
// RUN: %cxx -c -DCOMPARE_ERROR=5e-1 %include %s -o %t_main_qaic.o
// RUN: %cxx %t_qaic.o %t_qaic.bin %t_main_qaic.o %t_pp.o %odla_link -lodla_qaic -o %t_qaic.exe
// RUN: ODLA_AIC_NUM_CORES=14 ODLA_FORCE_AIC_SIM=1 %t_qaic.exe | FileCheck %s

// CHECK: Result verified
// CHECK: person, pos:(93,191 371,277), score:0.99
// CHECK: dog, pos:(266,63 346,207), score:0.99
// CHECK: horse, pos:(135,395 351,604), score:0.99

// clang-format on
#include <algorithm>
#include <array>
#include <iostream>
#include <vector>
#include <cassert>

constexpr int N = 1;
constexpr int H = 416;
constexpr int W = 416;
constexpr int C = 3;

#include "test_util.h"

extern "C" {
void yolo_v3(const float input[N * H * W * C],
             float out_conv2d_59[1 * 255 * 13 * 13],
             float out_conv2d_67[1 * 255 * 26 * 26],
             float out_conv2d_75[1 * 255 * 52 * 52]);
}
extern std::vector<std::pair<std::string, std::array<float, 5>>>
post_process_nhwc(int orig_img_w, int orig_img_h, float bb13[1 * 13 * 13 * 255],
                  float bb26[1 * 26 * 26 * 255], float bb52[1 * 52 * 52 * 255]);

#ifndef COMPARE_ERROR
#define COMPARE_ERROR 1e-3
#endif

static const float input_nhwc[N * H * W * C]{
#include "input_nhwc.dat"
};

static const int out_dim0 = H / 32;
static const int out_dim1 = H / 16;
static const int out_dim2 = H / 8;
static const int out_ch = 255;

static_assert(out_dim0 == 13);
static const float output_ref_nhwc_13[out_dim0 * out_dim0 * out_ch]{
#include "bbox_13x13.dat"
};
static const float output_ref_nhwc_26[out_dim1 * out_dim1 * out_ch]{
#include "bbox_26x26.dat"
};
static const float output_ref_nhwc_52[out_dim2 * out_dim2 * out_ch]{
#include "bbox_52x52.dat"
};

extern std::vector<std::pair<std::string, std::array<float, 5>>>
post_process_nhwc(int img_w, int img_h, float bb13[1 * 13 * 13 * 255],
                  float bb26[1 * 26 * 26 * 255], float bb52[1 * 52 * 52 * 255]);

int main(int argc, char** argv) {
  const float* input = input_nhwc;
  const float* ref_out0 = output_ref_nhwc_13;
  const float* ref_out1 = output_ref_nhwc_26;
  const float* ref_out2 = output_ref_nhwc_52;
  auto begin_time = Now();

#ifdef USE_NCHW_DATA
  std::array<float, N * C * H * W> input_nchw;
  input = to_nchw(input_nhwc, input_nchw.data(), N, H, W, C);
#endif

  std::vector<float> out_13(N * 255 * 13 * 13);
  std::vector<float> out_26(N * 255 * 26 * 26);
  std::vector<float> out_52(N * 255 * 52 * 52);
  yolo_v3(input, out_13.data(), out_26.data(), out_52.data());

#ifdef USE_NCHW_DATA
  auto permute_2d = [](std::vector<float>* arg, int src_dim0, int src_dim1) {
    assert(arg->size() == src_dim0 * src_dim1);
    std::vector<float> buf(arg->size());
    for (int i = 0; i < src_dim1; ++i)
      for (int j = 0; j < src_dim0; ++j)
        buf[i * src_dim0 + j] = (*arg)[j * src_dim1 + i];
    arg->swap(buf);
  };
  permute_2d(&out_13, out_ch, out_dim0 * out_dim0);
  permute_2d(&out_26, out_ch, out_dim1 * out_dim1);
  permute_2d(&out_52, out_ch, out_dim2 * out_dim2);
#endif

  if (!(Verify(out_13.data(), ref_out0,
               sizeof(output_ref_nhwc_13) / sizeof(output_ref_nhwc_13[0]),
               COMPARE_ERROR) &&
        Verify(out_26.data(), ref_out1,
               sizeof(output_ref_nhwc_26) / sizeof(output_ref_nhwc_26[0]),
               COMPARE_ERROR) &&
        Verify(out_52.data(), ref_out2,
               sizeof(output_ref_nhwc_52) / sizeof(output_ref_nhwc_52[0]),
               COMPARE_ERROR))) {
    printf(" Failed\n");
    return 1;
  }
  printf("Result verified\n");

  constexpr int src_w = 640;
  constexpr int src_h = 424;
  auto ret = post_process_nhwc(src_w, src_h, out_13.data(), out_26.data(),
                               out_52.data());
  auto end_time = Now();
  auto t = GetDuration(begin_time, end_time);
  printf("Elapse time: %lf seconds/iter.\n", t);

  for (auto& obj : ret) {
    printf("%s, pos:(%g,%g %g,%g), score:%f\n", obj.first.c_str(),
           round(obj.second[0]), round(obj.second[1]), round(obj.second[2]),
           round(obj.second[3]), obj.second[4]);
  }

#ifdef ITER_TEST
  const int iter = ITER_TEST;
#else
  const int iter = 1;
#endif
  if (iter > 1) {
    auto begin_time = Now();
    for (int i = 1; i <= iter; ++i) {
      yolo_v3(input, out_13.data(), out_26.data(), out_52.data());
    }
    auto end_time = Now();
    auto t = GetDuration(begin_time, end_time) / iter;
    auto rate = 1.0 / t;
    printf("Elapse time: %lf seconds/iter, %f iters/sec (avg of %d iters)\n", t,
           rate, iter);
  }
  return 0;
}
