//===- tflite_convert.td -------------------------------------*- tblgen -*-===//
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

include "convert.td"

def TFLITE_Add : OpMapping<"Add", Add>;

def TFLITE_ADD : TFLITEExtension<"ADD"> {
  let param_name_ = "builtin_options_as_ADD";
}

def TFLITE_CONV_2D : TFLITEExtension<"CONV_2D"> {
  let param_name_ = "builtin_options_as_Conv2DOptions";
  let extension_attr_ = [
    ExtensionAttr<"dilation_h_factor", Integer, "1">,
    ExtensionAttr<"dilation_w_factor", Integer, "1">,
    ExtensionAttr<"fused_activation_function", EnumActivationType, "NONE">,
    ExtensionAttr<"padding", EnumPadding, "VALID">,
    ExtensionAttr<"stride_h", Integer, "1">,
    ExtensionAttr<"stride_w", Integer, "1">
  ];
}

def TFLITE_DEPTHWISE_CONV_2D : TFLITEExtension<"DEPTHWISE_CONV_2D"> {
  let param_name_ = "builtin_options_as_DepthwiseConv2DOptions";
  let extension_attr_ = [
    ExtensionAttr<"depth_multiplier", Integer, "1">,
    ExtensionAttr<"dilation_h_factor", Integer, "1">,
    ExtensionAttr<"dilation_w_factor", Integer, "1">,
    ExtensionAttr<"fused_activation_function", EnumActivationType, "NONE">,
    ExtensionAttr<"padding", EnumPadding, "VALID">,
    ExtensionAttr<"stride_h", Integer, "1">,
    ExtensionAttr<"stride_w", Integer, "1">
  ];
}

def TFLITE_MAX_POOL_2D : OpMapping<"MAX_POOL_2D", PoolingMax> {
  let param_name_ = "builtin_options_as_Pool2DOptions";
  let attr_mapping_ = [ AttributeMapping<"", "data_format", "NHWC"> ];
  let extension_attr_ = [
    ExtensionAttr<"filter_height", Integer, "1">,
    ExtensionAttr<"filter_width", Integer, "1">,
    ExtensionAttr<"fused_activation_function", EnumActivationType, "NONE">,
    ExtensionAttr<"padding", EnumPadding, "VALID">,
    ExtensionAttr<"stride_h", Integer, "1">,
    ExtensionAttr<"stride_w", Integer, "1">
  ];
}

def TFLITE_AVERAGE_POOL_2D : TFLITEExtension<"AVERAGE_POOL_2D"> {
  let param_name_ = "builtin_options_as_Pool2DOptions";
  let attr_mapping_ = [ AttributeMapping<"", "data_format", "NHWC"> ];
  let extension_attr_ = [
    ExtensionAttr<"filter_height", Integer, "1">,
    ExtensionAttr<"filter_width", Integer, "1">,
    ExtensionAttr<"fused_activation_function", EnumActivationType, "NONE">,
    ExtensionAttr<"padding", EnumPadding, "VALID">,
    ExtensionAttr<"stride_h", Integer, "1">,
    ExtensionAttr<"stride_w", Integer, "1">
  ];
}

def TFLITE_CONCATENATION : OpMapping<"CONCATENATION", Concat> {
  let param_name_ = "builtin_options_as_ConcatenationOptions";
  let extension_attr_ = [
    ExtensionAttr<"fused_activation_function", EnumActivationType, "NONE">
  ];
}

def TFLITE_FULLY_CONNECTED : OpMapping<"FULLY_CONNECTED", MatMul> {
  let param_name_ = "builtin_options_as_FullyConnectedOptions";
 let extension_attr_ = [
    ExtensionAttr<"fused_activation_function", EnumActivationType, "NONE">,
    ExtensionAttr<"keep_num_dims", Bool, "false">,
    ExtensionAttr<"weights_format", String, "\"default\"">,
  ];
}

def TFLITE_RELU : OpMapping<"RELU", Relu>;

def TFLITE_RELU6 : OpMapping<"RELU6", Relu6>;

def TFLITE_SQUEEZE : TFLITEExtension<"SQUEEZE"> {
  let param_name_ = "builtin_options_as_SqueezeOptions";
  let extension_attr_ = [ ExtensionAttr<"squeeze_dims", IntegerList, "{}"> ];
}

def TFLITE_TANH: OpMapping<"TANH", Tanh>;

def TFLITE_RESHAPE : OpMapping<"RESHAPE", Reshape> {
  let param_name_ = "builtin_options_as_ReshapeOptions";
  let extension_attr_ = [ ExtensionAttr<"new_shape", IntegerList, "{}"> ];
}

def TFLITE_SOFTMAX : OpMapping<"SOFTMAX", Softmax> {
  let param_name_ = "builtin_options_as_SoftmaxOptions";
  let extension_attr_ = [ ExtensionAttr<"beta", Integer, "1"> ];
}

