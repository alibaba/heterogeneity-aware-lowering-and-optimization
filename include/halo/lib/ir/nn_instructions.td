//===- nn_instructions.td ------------------------------------*- tblgen -*-===//
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

let cat_ = cat_nn in {
  def LRN : Inst<"Local response normalization."> {
    let attrs_ = [
         Attr<"the size of local region.", Integer, "size">,
         Attr<"scaling parameter.", Float, "alpha", "1">,
         Attr<"exponent.", Float, "beta", "0.5">,
         Attr<"bias.", Float, "bias", "1.0">,
         Attr<"Input data format.", 
              EnumDataFormat, "data_format", "NCHW">
     ];

     let ins_ = [Arg<"The input image, shape is defined by data_format",
                    ArgType<[F16,F32]>, 4D> ];
     let outs_ = [Arg<"The result.", MatchArgType<0> >];
  }

  def BatchNorm : Inst<"Batch normalization."> {
    let attrs_ = [Attr<"Offset coefficient, omitted if X4 exists.",
                       Float, "offset">,
                  Attr<"Scale coefficient, omitted if X5 exists.",
                       Float, "scale">,
                  Attr<"Use to avoid division by zero.",
                       Float, "epsilon", "0.00001">,
                  Attr<"Input data format.",
                       EnumDataFormat, "data_format", "NHWC">,
                  Attr<"Pre-scale coefficient to mean and variance.",
                       Float, "pre_scaling_factor", "1.0">];
    let ins_ = [Arg<"The input image, shape is defined by data_format",
                    ArgType<[F16,F32]>, 4D>,
                Arg<"The mean NDArray", MatchArgType<0>, 4D>,
                Arg<"The variance NDArray", MatchArgType<0>, 4D>,
                OptionalArg<"The offset NDArray", MatchArgType<0>, 4D>,
                OptionalArg<"The scale NDArray", MatchArgType<0>, 4D>];
    let outs_ = [Arg<"The result.", MatchArgType<0> >];
  }

}