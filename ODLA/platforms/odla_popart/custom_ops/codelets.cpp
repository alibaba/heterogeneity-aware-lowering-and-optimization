//===- codelets.cpp -------------------------------------------------------===//
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
  Input<Vector<unsigned>> selected_indices;
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
    int n = scores.size();
    *selected_num = 0;
    for (int i = n - 1, selected = n - 1; i >= 0; --i) {
      if (scores[selected_indices[i]] < score_threshold) continue;
      if (i == n - 1 ||
          !is_overlapping(selected_indices[selected], selected_indices[i])) {
        selected = i;
        selected_indices[*selected_num] = selected_indices[i];
        selected_info[*selected_num][0] = boxes[selected_indices[i]][0];
        selected_info[*selected_num][1] = boxes[selected_indices[i]][1];
        selected_info[*selected_num][2] = boxes[selected_indices[i]][2];
        selected_info[*selected_num][3] = boxes[selected_indices[i]][3];
        selected_info[*selected_num][4] = scores[selected_indices[i]];
        (*selected_num)++;
      }
    }
    return true;
  }
};