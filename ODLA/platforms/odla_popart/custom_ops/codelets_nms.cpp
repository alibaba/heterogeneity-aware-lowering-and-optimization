//===- codelets_nms.cpp ---------------------------------------------------===//
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

#include <algorithm>
#include <numeric>
#include <poplar/Vertex.hpp>
using namespace poplar;

class NonMaxSuppressionVertex : public Vertex {
 public:
  Vector<Input<Vector<float>>> boxes;
  Input<Vector<float>> scores;
  Input<float> iou_threshold;
  Input<float> score_threshold;
  Input<Vector<unsigned>> perm;
  Vector<Output<Vector<float>>> selected_info;
  Output<unsigned> selected_num;
  bool is_overlapping(int selected, int curr) {
    auto y_top = std::max(boxes[selected][0], boxes[curr][0]);
    auto x_left = std::max(boxes[selected][1], boxes[curr][1]);
    auto y_bot = std::min(boxes[selected][2], boxes[curr][2]);
    auto x_right = std::min(boxes[selected][3], boxes[curr][3]);
    if (x_right < x_left || y_bot < y_top) return false;

    float comm_area = (x_right - x_left) * (y_bot - y_top);
    float selected_area = (boxes[selected][2] - boxes[selected][0]) *
                          (boxes[selected][3] - boxes[selected][1]);
    float curr_area =
        (boxes[curr][2] - boxes[curr][0]) * (boxes[curr][3] - boxes[curr][1]);

    float iou = comm_area / (selected_area + curr_area - comm_area);
    return iou > iou_threshold;
  };
  bool compute() {
    std::make_heap(perm.begin(), perm.end(), [&](unsigned a, unsigned b) {
      return scores[a] > scores[b];
    });
    std::sort_heap(perm.begin(), perm.end(), [&](unsigned a, unsigned b) {
      return scores[a] > scores[b];
    });
    int n = scores.size();
    *selected_num = 0;
    for (int i = 0, selected = 0; i < n; ++i) {
      if (scores[perm[i]] < score_threshold) continue;
      if (i == 0 || !is_overlapping(perm[selected], perm[i])) {
        selected = i;
        perm[*selected_num] = perm[i];
        selected_info[*selected_num][0] = boxes[perm[i]][0];
        selected_info[*selected_num][1] = boxes[perm[i]][1];
        selected_info[*selected_num][2] = boxes[perm[i]][2];
        selected_info[*selected_num][3] = boxes[perm[i]][3];
        selected_info[*selected_num][4] = scores[perm[i]];
        (*selected_num)++;
      }
    }
    return true;
  }
};

class DecodeVertex : public Vertex {
 public:
  Vector<InOut<Vector<float>>> boxes;
  Input<Vector<unsigned>> anchors;
  Input<unsigned> orig_img_w;
  Input<unsigned> orig_img_h;
  Input<unsigned> dim;
  Input<unsigned> grid_x;
  Input<unsigned> grid_y;

  bool compute() {
    unsigned num_anchors = 3;
    unsigned cls_num = boxes[0].size() - 5;
    float scale = std::min(416.0 / *orig_img_h, 416.0 / *orig_img_w);
    float new_shape_h = std::round(*orig_img_h * scale);
    float new_shape_w = std::round(*orig_img_w * scale);
    float offset_h = (416 - new_shape_h) / 2.0 / 416;
    float offset_w = (416 - new_shape_w) / 2.0 / 416;
    float scale_h = 416 / new_shape_h;
    float scale_w = 416 / new_shape_w;
    for (int a = 0; a < num_anchors; ++a) {
      auto box = boxes[a];
      auto dx = ((box[0] / (box[0] + 1)) + *grid_x) / *dim; // dx
      dx = (dx - offset_w) * scale_w;
      float dy = ((box[1] / (box[1] + 1)) + *grid_y) / *dim; // dy
      dy = (dy - offset_h) * scale_h;

      float dw = (box[2] * anchors[a << 1]) / 416 * scale_w;
      float dh = (box[3] * anchors[a << 1 | 1]) / 416 * scale_h;

      box[0] = (dy - dh / 2.0) * *orig_img_h; // y_min
      box[1] = (dx - dw / 2.0) * *orig_img_w; // x min
      box[2] = (dy + dh / 2.0) * *orig_img_h; // y_max
      box[3] = (dx + dw / 2.0) * *orig_img_w; // x max

      float confidence = (box[4]) / (box[4] + 1);
      box[4] = confidence;

      for (int c = 0; c < cls_num; ++c) {
        box[5 + c] = (box[5 + c]) / ((box[5 + c]) + 1) * confidence;
      }
    }

    return true;
  }
};
