//===- yolov3_postproc.cc -------------------------------------------------===//
//
// Copyright (C) 2019-2020 Alibaba Group Holding Limited.
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

#include <dlfcn.h>
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <iostream>
#include <popart/builder.hpp>
#include <popart/devicemanager.hpp>
#include <popart/logging.hpp>
#include <popart/ndarraywrapper.hpp>
#include <popart/op.hpp>
#include <popart/op/l1.hpp>
#include <popart/opmanager.hpp>
#include <popart/popx/opx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/session.hpp>
#include <popart/tensordata.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/tensornames.hpp>
#include <string>
#include <vector>

using namespace popart;

static const std::vector<std::string> ClassesNames{
#include "coco_classes.txt"
};
static const std::vector<int> AllAnchors{
#include "yolo_anchors.txt"
};

template <int batch, int dim, int cls_num>
static void decode(int orig_img_w, int orig_img_h, float* bb,
                   const std::vector<int>& anchor_mask) {
  int num_anchors = anchor_mask.size();
  float scale = std::min(416.0 / orig_img_h, 416.0 / orig_img_w);
  float new_shape_h = std::round(orig_img_h * scale);
  float new_shape_w = std::round(orig_img_w * scale);
  float offset_h = (416 - new_shape_h) / 2.0 / 416;
  float offset_w = (416 - new_shape_w) / 2.0 / 416;
  float scale_h = 416 / new_shape_h;
  float scale_w = 416 / new_shape_w;

  assert(num_anchors == 3);
  std::vector<std::pair<int, int>> anchors;
  for (auto m : anchor_mask) {
    anchors.push_back({AllAnchors[m * 2], AllAnchors[m * 2 + 1]});
  }
// [batch, dim, dim, num_anchors, 5 + cls_num]
// dx/dy [:, :, :, :, 0:2]
// dw/dh [:, :, :, :, 2:4]
// conf: [:, :, :, :, 4:5]
// prob: [:, :, :, :, 5: ]
#pragma omp parallel for
  for (size_t i = 0, e = batch * dim * dim * num_anchors * (cls_num + 5);
       i != e; ++i) {
    bb[i] = std::exp(bb[i]);
  }
  float* ptr = bb;
  for (int i = 0; i != batch; ++i) {
    for (int grid_y = 0; grid_y < dim; ++grid_y) {
      for (int grid_x = 0; grid_x < dim; ++grid_x) {
        for (int a = 0; a < num_anchors; ++a) {
          auto dx = ((ptr[0] / (ptr[0] + 1)) + grid_x) / dim; // dx
          dx = (dx - offset_w) * scale_w;
          float dy = ((ptr[1] / (ptr[1] + 1)) + grid_y) / dim; // dy
          dy = (dy - offset_h) * scale_h;

          float dw = (ptr[2] * anchors[a].first) / 416 * scale_w;
          float dh = (ptr[3] * anchors[a].second) / 416 * scale_h;

          *ptr++ = (dy - dh / 2.0) * orig_img_h; // y_min
          *ptr++ = (dx - dw / 2.0) * orig_img_w; // x min
          *ptr++ = (dy + dh / 2.0) * orig_img_h; // y_max
          *ptr++ = (dx + dw / 2.0) * orig_img_w; // x max

          float confidence = (*ptr) / ((*ptr) + 1);
          *ptr++ = confidence;

          for (int c = 0; c < cls_num; ++c, ++ptr) {
            *ptr = (*ptr) / ((*ptr) + 1) * confidence;
          }
        }
      }
    }
  }
}
// Using Custom Op. Too Slow...
std::vector<std::pair<std::string, std::array<float, 5>>> post_process_nhwc(
    int orig_img_w, int orig_img_h, float bb13[1 * 13 * 13 * 255],
    float bb26[1 * 26 * 26 * 255], float bb52[1 * 52 * 52 * 255]) {
  assert(ClassesNames.size() == 80);
  const std::vector<std::vector<int>> anchor_masks{
      {6, 7, 8}, {3, 4, 5}, {0, 1, 2}};
  decode<1, 13, 80>(orig_img_w, orig_img_h, bb13, anchor_masks[0]);
  decode<1, 26, 80>(orig_img_w, orig_img_h, bb26, anchor_masks[1]);
  decode<1, 52, 80>(orig_img_w, orig_img_h, bb52, anchor_masks[2]);
  // NMS
  float score_thre = 0.3;
  float iou_thre = 0.45;
  std::vector<std::pair<std::string, std::array<float, 5>>> ret;

  void* handle =
      dlopen("vision/detection/yolo/build/libcustom_ops.so", RTLD_LAZY);
  if (!handle) {
    std::cerr << "Cannot open library: " << dlerror() << std::endl;
  }

  const unsigned n1 = 13 * 13 * 3;
  const unsigned n2 = 26 * 26 * 3;
  const unsigned n3 = 52 * 52 * 3;
  const unsigned n = n1 + n2 + n3;
  auto builder = popart::Builder::create();
  auto aiOnnx = builder->aiOnnxOpset9();
  // Add input tensors
  popart::TensorInfo inputInfo1{"FLOAT", std::vector<int64_t>{n1, 85}};
  popart::TensorInfo inputInfo2{"FLOAT", std::vector<int64_t>{n2, 85}};
  popart::TensorInfo inputInfo3{"FLOAT", std::vector<int64_t>{n3, 85}};
  popart::TensorInfo inputInfo4{"FLOAT", std::vector<int64_t>{}};
  popart::TensorInfo inputInfo5{"FLOAT", std::vector<int64_t>{}};
  std::cout << "Adding input tensor\n";
  auto input1 = builder->addInputTensor(inputInfo1);
  auto input2 = builder->addInputTensor(inputInfo2);
  auto input3 = builder->addInputTensor(inputInfo3);
  auto inputbb = aiOnnx.concat({input1, input2, input3}, 0);
  auto input4 = builder->addInputTensor(inputInfo4);
  auto input5 = builder->addInputTensor(inputInfo5);

  // Add operation
  std::cout << "Adding custom operation nms(input)\n";
  const popart::OperatorIdentifier nms(popart::Domain::ai_graphcore, "NMS", 1,
                                       3, 2);
  auto o = builder->customOp(nms, 1, {inputbb, input4, input5}, 2, {});
  auto o1 = o[0];
  auto o2 = o[1];

  // Add output tensor
  std::cout << "Adding output tensor o\n";
  builder->addOutputTensor(o1);
  builder->addOutputTensor(o2);

  std::cout << "Getting model proto\n";
  auto proto = builder->getModelProto();

  std::cout << "Constructing DataFlow\n";
  auto dataFlow = popart::DataFlow(1, {{o1, popart::AnchorReturnType("ALL")},
                                       {o2, popart::AnchorReturnType("ALL")}});

  std::map<std::string, std::string> deviceOpts{{"numIPUs", "1"},
                                                {"tilesPerIPU", "40"}};
  auto ipuModelDevice =
      popart::DeviceManager::createDeviceManager().createIpuModelDevice(
          deviceOpts);
  // or acquireAvailableDevice();
  std::cout << "Creating session from Onnx Model...\n";
  auto session = popart::InferenceSession::createFromOnnxModel(proto, dataFlow,
                                                               ipuModelDevice);
  std::cout << "Creating session from Onnx Model...done\n";
  Shape zeroDim{};
  popart::NDArrayWrapper<float> inData1(bb13, {n1, 85});
  popart::NDArrayWrapper<float> inData2(bb26, {n2, 85});
  popart::NDArrayWrapper<float> inData3(bb52, {n3, 85});
  popart::NDArrayWrapper<float> inData4(&iou_thre, zeroDim);
  popart::NDArrayWrapper<float> inData5(&score_thre, zeroDim);
  std::map<popart::TensorId, popart::IArray&> inputs = {{input1, inData1},
                                                        {input2, inData2},
                                                        {input3, inData3},
                                                        {input4, inData4},
                                                        {input5, inData5}};
  // Prepare output tensor
  std::vector<unsigned> selected_num(80);
  std::vector<float> selected_info(80 * n * 5);
  popart::NDArrayWrapper<float> outData1(selected_info.data(), {80, n, 5});
  popart::NDArrayWrapper<uint32_t> outData2(selected_num.data(), {80});
  std::map<popart::TensorId, popart::IArray&> anchors = {{o1, outData1},
                                                         {o2, outData2}};
  std::cout << "Preparing session device...\n";
  session->prepareDevice();
  std::cout << "Preparing session device...done\n";

  popart::StepIO stepio(inputs, anchors);

  std::cout << "Running..."
            << "\n";
  session->run(stepio);
  std::cout << "Running...done"
            << "\n";
  for (int cls = 0, idx = 0; cls < ClassesNames.size(); ++cls, idx += n * 5) {
    for (int i = 0, e = selected_num[cls]; i < e; ++i) {
      int tmp = idx + i * 5;
      ret.push_back(
          {ClassesNames[cls],
           {selected_info[tmp], selected_info[tmp + 1], selected_info[tmp + 2],
            selected_info[tmp + 3], selected_info[tmp + 4]}});
    }
  }
  return ret;
}