//===- tf_convert.td -----------------------------------------*- tblgen -*-===//
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

// Define TF extension inst
def TF_Broadcast : TFExtension<"Broadcast">;
def TF_BroadcastTo : TFExtension<"BroadcastTo">;

def TF_Cast : TFExtension<"Cast"> {
  let extension_attr_ = [ExtensionAttr<"SrcT", EnumDataType, "INVALID">,
                         ExtensionAttr<"DstT", EnumDataType, "INVALID">,
                         ExtensionAttr<"Truncate", Bool, "false">];
}

def TF_Cos : OpMapping<"Cos", Cos>;

def TF_ExpandDims : TFExtension<"ExpandDims">;

def TF_Fill : TFExtension<"Fill">;

def TF_Identity : TFExtension<"Identity">;

def TF_LogSoftmax : OpMapping<"LogSoftmax", LogSoftmax> {
 let attr_mapping_ = [
   AttributeMapping<"axis", "axis", "-1">];
}

def TF_Reshape : TFExtension<"Reshape"> {
  let extension_attr_ = [ ExtensionAttr<"shape", IntegerList, "{}"> ];
}

def TF_Shape: OpMapping<"Shape", Shape> {
  let attr_mapping_ = [
    AttributeMapping<"", "data_type", "INT32">];
}

def TF_Squeeze : TFExtension<"Squeeze"> {
  let extension_attr_ = [ ExtensionAttr<"squeeze_dims", IntegerList, "{}"> ];
}

def TF_Split : TFExtension<"Split"> {
  let extension_attr_  = [ ExtensionAttr<"num_split", Integer, "0">];
  let num_outputs_ = -1;
}

def TF_StopGradient : TFExtension<"StopGradient">;

def TF_StridedSlice : TFExtension<"StridedSlice"> {
  let extension_attr_ = [ExtensionAttr<"begin_mask", Integer, "0">,
                         ExtensionAttr<"end_mask", Integer, "0">,
                         ExtensionAttr<"ellipsis_mask", Integer, "0">,
                         ExtensionAttr<"new_axis_mask", Integer, "0">,
                         ExtensionAttr<"shrink_axis_mask", Integer, "0">];
}

def TF_Sum : OpMapping<"Sum", ReduceSum> {
  let attr_mapping_ = [
    AttributeMapping<"keep_dims", "keep_dims", "false">
  ];
}

def TF_ResizeBilinear : OpMapping<"ResizeBilinear", Resize> {
  let attr_mapping_ = [
    AttributeMapping<"", "interpolation", "LINEAR">,
    AttributeMapping<"", "mode", "HALF_PIXEL_TF">,
    AttributeMapping<"", "axes_mask", "0x6"> // NHWC
    ];
}

def TF_FIFOQueueV2 : TFExtension<"FIFOQueueV2">;

def TF_QueueDequeueV2 : TFExtension<"QueueDequeueV2">;

def TF_SpaceToBatchND : TFExtension<"SpaceToBatchND">;

def TF_BatchToSpaceND : TFExtension<"BatchToSpaceND">;

// Define Halo instruction
def TF_Add : OpMapping<"Add", Add>;
def TF_AddV2 : OpMapping<"AddV2", Add>;
def TF_AddN : TFExtension<"AddN">;

def TF_ArgMax : OpMapping<"ArgMax", Argmax> {
 let attr_mapping_ = [
   AttributeMapping<"", "keep_dims", "false">];
}

def TF_And : OpMapping<"LogicalAnd", And>;

def TF_AvgPool : OpMapping<"AvgPool", PoolingAvg> {
  let attr_mapping_ = [
    AttributeMapping<"strides", "strides", "{1,1,1,1}">,
    AttributeMapping<"ksize", "ksize", "{1,1,1,1}">,
    AttributeMapping<"padding", "padding", "VALID">,
    AttributeMapping<"data_format", "data_format", "NHWC">
  ];
}

def TF_BiasAdd : OpMapping<"BiasAdd", Add>;

def TF_BatchMatMul : OpMapping<"BatchMatMul", MatMul> {
  let attr_mapping_ = [AttributeMapping<"adj_x", "transpose_a", "false">,
                       AttributeMapping<"adj_y", "transpose_b", "false">];
}

def TF_BatchMatMulV2 : OpMapping<"BatchMatMulV2", MatMul> {
  let attr_mapping_ = [AttributeMapping<"adj_x", "transpose_a", "false">,
                       AttributeMapping<"adj_y", "transpose_b", "false">];
}

def TF_Bitcast : OpMapping<"Bitcast", Bitcast> {
  let attr_mapping_ = [
    AttributeMapping<"type", "data_type", "INVALID">
  ];
}

def TF_ConcatV2 : OpMapping<"ConcatV2", Concat> {
  let attr_mapping_ = [AttributeMapping<"N", "N", "0">];
}

def TF_Conv2D : OpMapping<"Conv2D", Conv2D> {
  let attr_mapping_ = [
    AttributeMapping<"padding", "padding", "VALID">,
    AttributeMapping<"strides", "strides", "{1,1,1,1}">,
    AttributeMapping<"data_format", "data_format", "NHWC">,
    AttributeMapping<"", "filter_format", "HWCN">,
    AttributeMapping<"dilations", "dilations", "{1,1,1,1}">,
    AttributeMapping<"", "group", "1">
  ];
}

def TF_DepthwiseConv2D : OpMapping<"DepthwiseConv2dNative", Conv2D> {
  let attr_mapping_ = [
    AttributeMapping<"padding", "padding", "VALID">,
    AttributeMapping<"strides", "strides", "{1,1,1,1}">,
    AttributeMapping<"data_format", "data_format", "NHWC">,
    AttributeMapping<"", "filter_format", "HWCN">,
    AttributeMapping<"dilations", "dilations", "{1,1,1,1}">,
    AttributeMapping<"", "group", "0">
  ];
}

def TF_EQ : OpMapping<"Equal", Cmp> {
  let attr_mapping_ = [AttributeMapping<"", "predicator", "EQ">];
}

def TF_NE : OpMapping<"NotEqual", Cmp> {
  let attr_mapping_ = [AttributeMapping<"", "predicator", "NE">];
}

def TF_Erf : OpMapping<"Erf", Erf>;

def TF_Exp : OpMapping<"Exp", Exp>;

def TF_Floor : OpMapping<"Floor", Floor>;

def TF_Round: OpMapping<"Round", Round>;

def TF_FusedBatchNorm : OpMapping<"FusedBatchNorm", BatchNorm> {
  let attr_mapping_ = [
    AttributeMapping<"offset", "offset", "0">,
    AttributeMapping<"scale", "scale", "1">,
    AttributeMapping<"epsilon", "epsilon", "0.00001">,
    AttributeMapping<"data_format", "data_format", "NHWC">
  ];
}

def TF_GatherV2 : OpMapping<"GatherV2", Gather> {
  let attr_mapping_ = [AttributeMapping<"", "axis", "0">];
}

def TF_Greater : OpMapping<"Greater", Cmp> {
  let attr_mapping_ = [AttributeMapping<"", "predicator", "GT">];
}

def TF_GreaterEqual : OpMapping<"GreaterEqual", Cmp> {
  let attr_mapping_ = [AttributeMapping<"", "predicator", "GE">];
}

def TF_IpuGelu : OpMapping<"IpuGelu", Gelu> {
  let attr_mapping_ = [AttributeMapping<"", "use_approx", "false">];
}

def TF_LeakyRelu : OpMapping<"LeakyRelu", LeakyRelu> {
  let attr_mapping_ = [AttributeMapping<"alpha", "alpha", "1.0">];
}

def TF_Less : OpMapping<"Less", Cmp> {
  let attr_mapping_ = [AttributeMapping<"", "predicator", "LT">];
}

def TF_ListDiff : OpMapping<"ListDiff", SetDiff1D>;

def TF_MatMul: OpMapping<"MatMul", MatMul> {
  let attr_mapping_ = [
    AttributeMapping<"transpose_a", "transpose_a", "false">,
    AttributeMapping<"transpose_b", "transpose_b", "false">
  ];
}

def TF_Max : OpMapping<"Max", ReduceMax> {
  let attr_mapping_ = [
    AttributeMapping<"keep_dims", "keep_dims", "false">,
    AttributeMapping<"axis", "axis", "{}">];
}

def TF_Maximum : OpMapping<"Maximum", Maximum>;
def TF_Minimum : OpMapping<"Minimum", Minimum>;

def TF_MaxPool : OpMapping<"MaxPool", PoolingMax> {
  let attr_mapping_ = [
    AttributeMapping<"strides", "strides", "{1,1,1,1}">,
    AttributeMapping<"ksize", "ksize", "{1,1,1,1}">,
    AttributeMapping<"padding", "padding", "VALID">,
    AttributeMapping<"data_format", "data_format", "NHWC">
  ];
}

def TF_Mean : OpMapping<"Mean", ReduceMean> {
  let attr_mapping_ = [
    AttributeMapping<"keep_dims", "keep_dims", "false">,
    AttributeMapping<"axis", "axis", "{}">
  ];
}

def TF_Merge : TFExtension<"Merge">;

def TF_Mul : OpMapping<"Mul", Mul>;

def TF_Neg : OpMapping<"Neg", Neg>;

def TF_NoOp : OpMapping<"NoOp", NoOp>;

def TF_OneHot : OpMapping<"OneHot", OneHot> {
  let attr_mapping_ = [
    AttributeMapping<"axis", "axis", "-1">];
}

def TF_Or : OpMapping<"LogicalOr", Or>;

def TF_Pad : OpMapping<"Pad", Pad> {
  let attr_mapping_ =
      [AttributeMapping<"mode", "mode", "CONSTANT"> ];
}

def TF_Pack : OpMapping<"Pack", Stack> {
  let attr_mapping_ = [
    AttributeMapping<"axis", "axis", "0">];
}

def TF_Pow : OpMapping<"Pow", Pow>;

def TF_Prod : OpMapping<"Prod", ReduceProduct> {
  let attr_mapping_ = [
    AttributeMapping<"keep_dims", "keep_dims", "false">,
    AttributeMapping<"axis", "axis", "{}">];
}

def TF_RealDiv : OpMapping<"RealDiv", Div>;

def TF_Relu : OpMapping<"Relu", Relu>;

def TF_Relu6 : OpMapping<"Relu6", Relu6>;

def TF_Rsqrt : OpMapping<"Rsqrt", Rsqrt>;

def TF_Select : OpMapping<"Select", Select>;

def TF_Sin : OpMapping<"Sin", Sin>;

def TF_Size : TFExtension<"Size">;

def TF_Slice : OpMapping<"Slice", Slice>;

def TF_Softmax : OpMapping<"Softmax", Softmax>;

def TF_Sqrt : OpMapping<"Sqrt", Sqrt>;

def TF_Square: TFExtension<"Square">;

def TF_Sub : OpMapping<"Sub", Sub>;

def TF_Tanh : OpMapping<"Tanh", Tanh>;

def TF_Transpose : OpMapping<"Transpose", Transpose> {
  let attr_mapping_ = [
    AttributeMapping<"perm", "permutation", "{}">];
}

def TF_RandomUniform : OpMapping<"RandomUniform", RandomUniform> {
  let attr_mapping_ = [
    AttributeMapping<"seed", "seed", "0">,
    AttributeMapping<"dtype", "dtype", "FLOAT32">,
    AttributeMapping<"", "minval", "0.0">,
    AttributeMapping<"", "maxval", "1.0">];
}

def TF_HgEngine : OpMapping<"HgEngine", HgEngine> {
  let attr_mapping_ = [
    AttributeMapping<"serialized_engine", "serialized_engine", "{}">,
    AttributeMapping<"in_data_format", "in_data_format", "\"NHWC\"">,
    AttributeMapping<"out_data_format", "out_data_format", "\"NHWC\"">,
    AttributeMapping<"in_binding_list", "in_binding_list", "{}">,
    AttributeMapping<"out_binding_list", "out_binding_list", "{}">,
    AttributeMapping<"output_shapes", "output_shapes", "{}">,
    AttributeMapping<"device_id", "device_id", "-1">
  ];
}

def TF_HgQuant : TFExtension<"HgQuant"> {
  let extension_attr_ = [
    ExtensionAttr<"in_scale", FloatList, "{1.0}">,
    ExtensionAttr<"in_bias", FloatList, "{0.0}">,
    ExtensionAttr<"qtype", String, "\"int8\"">,
    ExtensionAttr<"is_per_channel", Bool, "false">,
    ExtensionAttr<"round_mode", Integer, "0">,
    ExtensionAttr<"in_data_format", String, "\"NHWC\"">,
    ExtensionAttr<"out_data_format", String, "\"NHWC\"">,
    ExtensionAttr<"model_name", String, "\"\"">,
    ExtensionAttr<"op_name", String, "\"\"">,
    ExtensionAttr<"in_type", EnumDataType, "FLOAT32">,
    ExtensionAttr<"out_type", EnumDataType, "INT8">
  ];
}

def TF_HgDequant : TFExtension<"HgDequant"> {
  let extension_attr_ = [
    ExtensionAttr<"in_scale", FloatList, "{1.0}">,
    ExtensionAttr<"in_bias", FloatList, "{0.0}">,
    ExtensionAttr<"is_per_channel", Bool, "false">,
    ExtensionAttr<"in_data_format", String, "\"NHWC\"">,
    ExtensionAttr<"out_data_format", String, "\"NHWC\"">,
    ExtensionAttr<"model_name", String, "\"\"">,
    ExtensionAttr<"op_name", String, "\"\"">,
    ExtensionAttr<"in_type", EnumDataType, "INT8">,
    ExtensionAttr<"out_type", EnumDataType, "FLOAT32">
  ];
}

def TF_TopK  : OpMapping<"TopKV2", TopK>;

def TF_Range : OpMapping<"Range", Range>;

def TF_Sigmoid : OpMapping<"Sigmoid", Sigmoid>;

def TF_SquaredDifference : OpMapping<"SquaredDifference", SquaredDifference>;

def TF_Switch : TFExtension<"Switch"> { let num_outputs_ = 2; }

def TF_Tile : OpMapping<"Tile", Tile>;

def TF_Unique : OpMapping<"Unique", Unique> {
  let attr_mapping_ = [
    AttributeMapping<"out_idx", "out_idx_type", "INT32">];
}

def TF_ZerosLike : TFExtension<"ZerosLike"> {
  let extension_attr_ = [
   ExtensionAttr<"dtype", EnumDataType, "INVALID">
  ];
}
