//===- object_detection_instructions.td ----------------------*- tblgen -*-===//
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

#ifdef INSTRUCTION_BASE
#else
include "instruction_base.td"
#endif

let cat_ = cat_objectDetection in {

  def NonMaxSuppression : Inst<"Select a subset of bounding box in descending"
                               " order of scores."> {
    let attrs_ = [
      Attr<"Coefficient to adjust num_threshold in a fast nms algorithm.",
           Float, "eta", "1.0">,
      Attr<"The min score for a box to be selected.", Float, "score_threshold",
           "-1.0">,
      Attr<"The datatype of the result indices", EnumDataType, "index_type", "INT32">,
    ];
    let ins_ = [
      Arg<"Input box, 2-D of [num_boxes, 4].", ArgType<[ F32 ]>, 2D>,
      Arg<"Input score, 1-D.", MatchArgType<0>, 1D>,
      Arg<"Maximum number of boxes in the result", ArgType<[ I32 ]>, 0D>,
      Arg<"Input iou_threshold, the intersection over union threshold used to "
          "suppress boxes",
          ArgType<[ F32 ]>, 0D>,
      Arg<"Input score_threshold, the intersection over union threshold used "
          "to suppress scores",
          ArgType<[ F32 ]>, 0D>
    ];
    let outs_ = [Arg<"The result of the selected indices from input box, 1-D.",
                     ArgType<[I32, I64]>, 1D>];
  }

  def TopK : Inst <"Calculate values and indices of the largest k elements of"
                   " input along the last dimension."> {
    let attrs_ = [
      Attr<"Dimension on which to do the sort.", Integer, "axis", "-1">,
      Attr<"Whether to return the top-K largest or smallest elements.", Bool,
           "largest", "true">,
      Attr<"Whether the input is sorted.", Bool, "sorted", "true">,
      Attr<"The datatype of the result indices", EnumDataType, "index_type", "INT32">,
    ];
    let ins_ = [Arg<"Input data.", ArgType<[I8, I16, I32, F16, F32]> >,
                Arg<"Input k.", ArgType<[I32]>, 0D>];
    let outs_ = [Arg<"The result value.", MatchArgType<0> >,
                 Arg<"The result indices.", ArgType<[I32, I64]> >];
  }

  def Resize : Inst<"Resize the input tensor."> {
    let attrs_ = [Attr<"Coordinating mode such as HALF_PIXEL, "
                       "TF_HALF_PIXEL, ALIGN_CORNETS, ASYMMETRIC",
                        EnumResizeMode, "mode", "HALF_PIXEL">,
                  Attr<"Interpolation mode such as NEAREST, LINEAR and CUBIC",
                        EnumInterpolation, "interpolation_mode", "NEAREST">,
                  Attr<"The mask for axes to be resized", Integer, "axes_mask", "0">,
                  Attr<"Explicit result shape", Bool, "explicit_shape", "true">
                 ];
    let ins_ = [Arg<"Input data.", ArgType<[I8, I16, I32, F16, F32]> >,
                Arg<"Resized shape", ArgType<[I64, F32]> >];
    let outs_ = [Arg<"The result value.", MatchArgType<0>>];
  }

}