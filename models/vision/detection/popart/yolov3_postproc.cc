//===- yolov3_postproc.cc -------------------------------------------------===//
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

// RUN: echo ""
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <ctime>
#include <iostream>
#include <string>
#include <vector>

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

std::vector<std::pair<std::string, std::array<float, 5>>> post_process_nhwc(
    int orig_img_w, int orig_img_h, float bb13[1 * 13 * 13 * 255],
    float bb26[1 * 26 * 26 * 255], float bb52[1 * 52 * 52 * 255]) {
  auto begin = clock();
  assert(ClassesNames.size() == 80);
  const std::vector<std::vector<int>> anchor_masks{
      {6, 7, 8}, {3, 4, 5}, {0, 1, 2}};
  decode<1, 13, 80>(orig_img_w, orig_img_h, bb13, anchor_masks[0]);
  decode<1, 26, 80>(orig_img_w, orig_img_h, bb26, anchor_masks[1]);
  decode<1, 52, 80>(orig_img_w, orig_img_h, bb52, anchor_masks[2]);

  // NMS
  constexpr float score_thre = 0.9;
  constexpr float iou_thre = 0.45;
  std::vector<std::pair<std::string, std::array<float, 5>>> ret;
  for (int cls = 0; cls < ClassesNames.size(); ++cls) {
    std::vector<const float*> boxes;
    boxes.reserve((13 * 13 + 26 * 26 + 52 * 52) * 3);
    auto append = [&boxes](const float* start, int n, int cls, float thre) {
      for (int i = 0; i < n; ++i) {
        if (start[5 + cls] >= score_thre) {
          boxes.push_back(start);
        }
        start += 85;
      }
    };
    append(bb13, 13 * 13 * 3, cls, score_thre);
    append(bb26, 26 * 26 * 3, cls, score_thre);
    append(bb52, 52 * 52 * 3, cls, score_thre);
    std::sort(boxes.begin(), boxes.end(),
              [cls](const float* lhs, const float* rhs) {
                return lhs[5 + cls] > rhs[5 + cls];
              });
    auto is_overlapping = [&boxes, iou_thre](const float* selected,
                                             const float* curr) {
      // assume origin is top left.
      auto y_top = std::max(selected[0], curr[0]);
      auto x_left = std::max(selected[1], curr[1]);
      auto y_bot = std::min(selected[2], curr[2]);
      auto x_right = std::min(selected[3], curr[3]);
      if (x_right < x_left || y_bot < y_top) {
        return false;
      }

      float comm_area = (x_right - x_left) * (y_bot - y_top);
      float selected_area =
          (selected[2] - selected[0]) * (selected[3] - selected[1]);
      float curr_area = (curr[2] - curr[0]) * (curr[3] - curr[1]);

      float iou = comm_area / (selected_area + curr_area - comm_area);
      return iou > iou_thre;
    };

    for (int i = 0, selected = 0, e = boxes.size(); i < e; ++i) {
      if (i == 0 || !is_overlapping(boxes[selected], boxes[i])) {
        selected = i;
        ret.push_back({ClassesNames[cls],
                       {boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3],
                        boxes[i][5 + cls]}});
      }
    }
  }
  auto end = clock();
  printf("Post process %lfs.\n", (end - begin) * 1.0 / CLOCKS_PER_SEC);
  return ret;
}
