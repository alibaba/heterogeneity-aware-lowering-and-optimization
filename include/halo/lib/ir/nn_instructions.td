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

  def GroupNorm : Inst<"Group normalization."> {
    let attrs_ = [
      Attr<"Use to avoid division by zero.", Float, "epsilon", "0.00001">,
      Attr<"Input data format.", EnumDataFormat, "data_format", "NHWC">,
      Attr<"Number of groups", Integer, "groups", "-1">
    ];
    let ins_ = [
      Arg<"The input image, shape is defined by data_format",
          ArgType<[ F16, F32 ]>, 4D>,
      Arg<"The scale Tensor", ArgType<[F16, F32]>, 4D>,
      Arg<"The bias Tensor", ArgType<[F16, F32]>, 4D>
    ];
    let outs_ = [Arg<"The result.", MatchArgType<0> >];
  }

  def InstanceNorm : Inst<"Instance normalization."> {
    let attrs_ = [
      Attr<"Use to avoid division by zero.", Float, "epsilon", "0.00001">,
      Attr<"Input data format.", EnumDataFormat, "data_format", "NHWC">
    ];
    let ins_ = [
      Arg<"The input image, shape is defined by data_format",
          ArgType<[ F16, F32 ]>, 4D>,
      Arg<"The scale Tensor", ArgType<[F16, F32]>, 4D>,
      Arg<"The bias Tensor", ArgType<[F16, F32]>, 4D>
    ];
    let outs_ = [Arg<"The result.", MatchArgType<0> >];
  }

  def LpNormalize : Inst<"Lp Normalization."> {
    let attrs_ = [Attr<"Axis along which to normalize.",
                   IntegerList, "axis">,
                  Attr<"The exponent value in the norm formulation.",
                       Integer, "p", "2">,
                  Attr<"Use to avoid division by zero.",
                       Float, "epsilon", "1e-10">,
                  Attr<"Input data format.",
                       EnumDataFormat, "data_format", "NCHW">];
    let ins_ = [Arg<"The input tensor, shape is defined by data_format",
                    ArgType<[F16,F32]> >,
                OptionalArg<"The scale NDArray", MatchArgType<0> >];
    let outs_ = [Arg<"The result.", MatchArgType<0> >];
  }

  def LayerNorm : Inst<"Layer Normalization."> {
    let attrs_ = [Attr<"Axis along which to normalize.",
                   IntegerList, "axis", "{-1}">,
                  Attr<"Use to avoid division by zero.",
                       Float, "epsilon", "1e-12">];
    let ins_ = [Arg<"The input tensor, shape is usually (batch, sentence_length, embedding_dim).",
                    ArgType<[F16,F32]> >,
                Arg<"The scale Tensor(gamma)", ArgType<[F16, F32]>>,
                Arg<"The offset Tensor(beta)", ArgType<[F16, F32]>>];
    let outs_ = [Arg<"The result.", MatchArgType<0> >];
  }

  def TFIDFVectorize : Inst<"TF-IDF Text Vectorization."> {
    let attrs_ = [
      Attr<"Mininum n-gram length.", Integer, "min_gram_length", "1">,
      Attr<"Maximum n-gram length.", Integer, "max_gram_length", "1">,
      Attr<"Maximum number of words to be skipped.", Integer, "max_skip", "0">,
      Attr<"Maximum index", Integer64, "max_idx", "0">,
      Attr<"Weighting mode", EnumTFIDF, "mode", "INVALID">,
    ];
    let ins_ = [
      Arg<"The input for n-gram extraction",
          ArgType<[ Integer, Integer64, String ]>>,
      Arg<"The n-gram pool", MatchArgType<0>>,
      Arg<"Starting indices of n-grams", ArgType<[I64]>, 1D>,
      Arg<"Index mapping for output", ArgType<[I64]>, 1D>,
      OptionalArg<"Weights", ArgType<[F32]>, 1D>,
    ];
    let outs_ = [Arg<"The result.", ArgType<[Float]>>];
  }

  def MultiHeadAttention : Inst<"Multi-Head Attention."> {
    let attrs_ = [
      Attr<"Number of heads.", Integer, "num_heads", "0">,
      Attr<"Hidden size of each head.", Integer, "head_size", "0">,
      Attr<"Value for unused position.", Float, "filler", "-10000.0">,
      Attr<"Use layer norm.", Bool, "use_layer_norm", "true">,
    ];
    let ins_ = [
      Arg<"The input embedding", ArgType<[ Integer, Integer64, Float ]>>,
      Arg<"Q weight (transposed)", MatchArgType<0>>,
      Arg<"Q bias", MatchArgType<0>>,
      Arg<"K weight (transposed)", MatchArgType<0>>,
      Arg<"K bias", MatchArgType<0>>,
      Arg<"V weight (transposed)", MatchArgType<0>>,
      Arg<"V bias", MatchArgType<0>>,
      OptionalArg<"mask", ArgType<[ Integer, Integer64]>>,
    ];
    let outs_ = [Arg<"The result.", MatchArgType<0> >];
  }
}
