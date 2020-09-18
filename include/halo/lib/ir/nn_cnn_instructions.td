//===- nn_cnn_instructions.td --------------------------------*- tblgen -*-===//
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

let cat_ = cat_nn_cnn in {

  def Conv2D : Inst<"Perform 2D convolution on input X1 with filter X2."> {
    let attrs_ = [Attr<"The padding algoririthm.",
                       EnumPadding, "padding", "VALID">,
                  Attr<"The sliding window for each dimension of input, its "
                       "order is determined by data_format.",
                       IntegerList, "strides", "{1,1,1,1}">,
                  Attr<"The input data format.",
                       EnumDataFormat, "data_format", "NHWC">,
                  Attr<"The filter data format.",
                       EnumDataFormat, "filter_format", "NHWC">,
                  Attr<"The dilation factor for each dimension of input, its "
                       "order is determined by data_format.",
                       IntegerList, "dilations", "{1,1,1,1}">,
                  Attr<"The explicit padding to the left of the input.",
                       Integer, "padding_left", "0">,
                  Attr<"The explicit padding to the right of the input.",
                       Integer, "padding_right", "0">,
                  Attr<"The explicit padding to the top of the input.",
                       Integer, "padding_top", "0">,
                  Attr<"The explicit padding to the bottom of the input.",
                       Integer, "padding_bottom", "0">,
                  Attr<"The group size for depthwise conv",
                       Integer, "group", "1">];
    let ins_ = [Arg<"The input image.", ArgType<[F16,F32]>, 4D>,
                Arg<"The filter.", MatchArgType<0>, 4D>];
    let outs_ = [Arg<"The result.", MatchArgType<0>, 4D>];
  }

  def Conv2DTranspose: Inst <"Convolve input X1 with filter X2 by taking"
                             " opposite computation meaning with the filter"
                             " and attributes."> {
    let attrs_ = [Attr<"The padding algoririthm.",
                       EnumPadding, "padding", "VALID">,
                  Attr<"The sliding window for each dimension of input, its "
                       "order is determined by data_format.",
                       IntegerList, "strides", "{1,1,1,1}">,
                  Attr<"The input data format.",
                       EnumDataFormat, "data_format", "NHWC">,
                  Attr<"The filter data format.",
                       EnumDataFormat, "filter_format", "NHWC">,
                  Attr<"The dilation factor for each dimension of input, its "
                       "order is determined by data_format.",
                       IntegerList, "dilations", "{1,1,1,1}">,
                  Attr<"The explicit padding to the left of the input.",
                       Integer, "padding_left", "0">,
                  Attr<"The explicit padding to the right of the input.",
                       Integer, "padding_right", "0">,
                  Attr<"The explicit padding to the top of the input.",
                       Integer, "padding_top", "0">,
                  Attr<"The explicit padding to the bottom of the input.",
                       Integer, "padding_bottom", "0">,
                  Attr<"The spatial kernel size along each dimension, its "
                       "order is based on the filter_format.",
                       IntegerList, "kernel_shape", "{}">,
                  Attr<"The output shape", IntegerList, "output_shape", "{}">,
                  Attr<"The group size for output channels are divided into.",
                       Integer, "group", "1">];
    let ins_ = [Arg<"The input image.", ArgType<[F16,F32]>, 4D>,
                Arg<"The filter.", MatchArgType<0>, 4D>];
    let outs_ = [Arg<"The result.", MatchArgType<0>, 4D>];
}

  def PoolingAvg : Inst<"Compute the down sampling on input such that"
                        " the average value of the subset is retrieved."> {
    let attrs_ = [Attr<"The stride values along each dimension of input,"
                       "its order is based on data_format.",
                       IntegerList, "strides", "{1,1,1,1}">,
                  Attr<"The spatial kernel size along each dimension, its "
                       "order is based on the data_format.",
                       IntegerList, "ksize", "{1,1,1,1}">,
                  Attr<"The padding algoririthm.",
                       EnumPadding, "padding", "VALID">,
                  Attr<"The explicit padding to the left of the input.",
                       Integer, "padding_left", "0">,
                  Attr<"The explicit padding to the right of the input.",
                       Integer, "padding_right", "0">,
                  Attr<"The explicit padding to the top of the input.",
                       Integer, "padding_top", "0">,
                  Attr<"The explicit padding to the bottom of the input.",
                       Integer, "padding_bottom", "0">,
                  Attr<"The input data format.",
                       EnumDataFormat, "data_format", "NHWC">,
                  Attr<"The constant padding value",
                       Float, "pad_value", "0.0">];
    let ins_ = [Arg<"The input image.",
                    ArgType<[I8, I16, I32, F16, F32]>, 4D>];
    let outs_ = [Arg<"The result.", MatchArgType<0>, 4D>];
  }

  def PoolingMax : Inst<"Compute the down sampling on input such that"
                        " the largest value of the subset is retrieved."> {
    let attrs_ = [Attr<"The stride values along each dimension of input,"
                       "its order is based on data_format.",
                       IntegerList, "strides", "{1,1,1,1}">,
                  Attr<"The spatial kernel size along each dimension, its "
                       "order is based on the data_format.",
                       IntegerList, "ksize", "{1,1,1,1}">,
                  Attr<"The padding algoririthm.",
                       EnumPadding, "padding", "VALID">,
                  Attr<"The explicit padding to the left of the input.",
                       Integer, "padding_left", "0">,
                  Attr<"The explicit padding to the right of the input.",
                       Integer, "padding_right", "0">,
                  Attr<"The explicit padding to the top of the input.",
                       Integer, "padding_top", "0">,
                  Attr<"The explicit padding to the bottom of the input.",
                       Integer, "padding_bottom", "0">,
                  Attr<"The input data format.",
                       EnumDataFormat, "data_format", "NHWC">,
                  Attr<"The constant padding value", Float, "pad_value", "0">];
    let ins_ = [Arg<"The input image.",
                    ArgType<[I8, I16, I32, F16, F32]>, 4D>];
    let outs_ = [Arg<"The result.", MatchArgType<0>, 4D>];
  }

}