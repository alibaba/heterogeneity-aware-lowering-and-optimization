//===- onnx_convert.td ---------------------------------------*- tblgen -*-===//
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

// TODO (unknown): GlobalAveragePool op averages data within the same channel.
// For ND image where N>2, axis = range(shape.size()-2, shape.size())
def ONNX_GlobalAveragePool : OpMapping<"GlobalAveragePool", ReduceMean> {
  let attr_mapping_ = [
    AttributeMapping<"", "axis", "{2,3}">,
    AttributeMapping<"", "keep_dims", "true">];
}

def ONNX_Add : OpMapping<"Add", Add>;
def ONNX_Div : OpMapping<"Div", Div>;

def ONNX_BatchNormalization : OpMapping<"BatchNormalization", BatchNorm> {
  let attr_mapping_ = [
    AttributeMapping<"epsilon", "epsilon", "0.00001">,
    AttributeMapping<"", "data_format", "NCHW">];
}

def ONNX_LRN : OpMapping<"LRN", LRN> {
  let attr_mapping_ = [
    AttributeMapping<"size", "size", "0">,
    AttributeMapping<"alpha", "alpha", "0.00001">,
    AttributeMapping<"beta", "beta", "0.75">,
    AttributeMapping<"bias", "bias", "1.0">,
    AttributeMapping<"", "data_format", "NCHW">
  ];
}

def ONNX_Conv : OpMapping<"Conv", Conv2D> {
  let attr_mapping_ = [
    AttributeMapping<"auto_pad", "padding", "EXPLICIT">,
    AttributeMapping<"dilations", "dilations", "{1,1}", -1, 1>,
    AttributeMapping<"strides", "strides", "{1,1}", -1, 1>,
    AttributeMapping<"", "data_format", "NCHW">,
    AttributeMapping<"", "filter_format", "NCHW">,
    AttributeMapping<"group", "group", "1">,
    AttributeMapping<"pads", "padding_left", "{0,0,0,0}", 1>,
    AttributeMapping<"pads", "padding_top", "{0,0,0,0}", 0>,
    AttributeMapping<"pads", "padding_right", "{0,0,0,0}", 3>,
    AttributeMapping<"pads", "padding_bottom", "{0,0,0,0}", 2>];
  let extension_attr_ = [
    ExtensionAttr<"kernel_shape", IntegerList, "{}">];
}

def ONNX_Gemm: OpMapping<"Gemm", Gemm> {
  let attr_mapping_ = [
    AttributeMapping<"alpha", "alpha", "1.0">,
    AttributeMapping<"beta", "beta", "1.0">,
    AttributeMapping<"transA", "transpose_a", "0">,
    AttributeMapping<"transB", "transpose_b", "0">];
}

def ONNX_MatMul: OpMapping<"MatMul", MatMul> {
  let attr_mapping_ = [
    AttributeMapping<"", "transpose_a", "false">,
    AttributeMapping<"", "transpose_b", "false">
  ];
}

def ONNX_Mul : OpMapping<"Mul", Mul>;

def ONNX_MaxPool : OpMapping<"MaxPool", PoolingMax> {
  let attr_mapping_ = [
    AttributeMapping<"auto_pad", "padding", "EXPLICIT">,
    AttributeMapping<"strides", "strides", "{1,1}", -1, 1>,
    AttributeMapping<"kernel_shape", "ksize", "{1,1}", -1, 1>,
    AttributeMapping<"", "data_format", "NCHW">,
    AttributeMapping<"pads", "padding_left", "{0,0,0,0}", 0>,
    AttributeMapping<"pads", "padding_top", "{0,0,0,0}", 1>,
    AttributeMapping<"pads", "padding_right", "{0,0,0,0}", 2>,
    AttributeMapping<"pads", "padding_bottom", "{0,0,0,0}", 3>,
    AttributeMapping<"group", "group", "1">];
  let extension_attr_ = [
    ExtensionAttr<"ceil_mode", Integer, "0">,
    ExtensionAttr<"dilations", IntegerList, "{1,1}">,
    ExtensionAttr<"storage_order", Integer, "0">];
}

def ONNX_AveragePool : OpMapping<"AveragePool", PoolingAvg> {
  let attr_mapping_ = [
    AttributeMapping<"auto_pad", "padding", "EXPLICIT">,
    AttributeMapping<"strides", "strides", "{1,1}", -1, 1>,
    AttributeMapping<"kernel_shape", "ksize", "{1,1}", -1, 1>,
    AttributeMapping<"", "data_format", "NCHW">,
    AttributeMapping<"pads", "padding_left", "{0,0,0,0}", 0>,
    AttributeMapping<"pads", "padding_top", "{0,0,0,0}", 1>,
    AttributeMapping<"pads", "padding_right", "{0,0,0,0}", 2>,
    AttributeMapping<"pads", "padding_bottom", "{0,0,0,0}", 3>,
    AttributeMapping<"group", "group", "1">];
  let extension_attr_ = [
    ExtensionAttr<"ceil_mode", Integer, "0">,
    ExtensionAttr<"dilations", IntegerList, "{1,1}">,
    ExtensionAttr<"storage_order", Integer, "0">];
}

def ONNX_Relu : OpMapping<"Relu", Relu>;

def ONNX_ReduceMin: OpMapping<"ReduceMin", ReduceMin> {
  let attr_mapping_ = [
    AttributeMapping<"keepdims", "keep_dims", "true">,
    AttributeMapping<"axes", "axis", "{}">
  ];
}

def ONNX_ReduceMean: OpMapping<"ReduceMean", ReduceMean> {
  let attr_mapping_ = [
    AttributeMapping<"keepdims", "keep_dims", "true">,
    AttributeMapping<"axes", "axis", "{}">
  ];
}

def ONNX_Concat : OpMapping<"Concat", Concat> {
  let attr_mapping_ = [AttributeMapping<"axis", "axis", "1">,
                       AttributeMapping<"", "N", "0">];
}

def ONNX_Clip : OpMapping<"Clip", Relu6>;

def ONNX_Exp : OpMapping<"Exp", Exp>;
def ONNX_Sqrt : OpMapping<"Sqrt", Sqrt>;
def ONNX_Floor : OpMapping<"Floor", Floor>;
def ONNX_Ceil : OpMapping<"Ceil", Ceil>;

def ONNX_Gather : OpMapping<"Gather", Gather> {
  let attr_mapping_ = [AttributeMapping<"axis", "axis", "0">];
}

def ONNX_Reshape : OpMapping<"Reshape", Reshape>;

def ONNX_Shape : ONNXExtension<"Shape">;

def ONNX_Softmax : OpMapping<"Softmax", Softmax>;

def ONNX_Squeeze : ONNXExtension<"Squeeze"> {
  let extension_attr_ = [ ExtensionAttr<"axes", IntegerList, "{}"> ];
}

def ONNX_Sub : OpMapping<"Sub", Sub>;

def ONNX_Transpose : OpMapping<"Transpose", Transpose> {
  let attr_mapping_ = [
    AttributeMapping<"perm", "permutation", "{}">];
}

def ONNX_NonMaxSupression : OpMapping<"NonMaxSuppression", NonMaxSuppression> {
  let attr_mapping_ = [
    AttributeMapping<"", "eta", "1.0">,
    AttributeMapping<"", "score_threshold", "-1.0">,
    AttributeMapping<"", "index_type", "INT64">];
}

def ONNX_TopK : OpMapping<"TopK", TopK> {
  let attr_mapping_ = [
    AttributeMapping<"axis", "axis", "-1">,
    AttributeMapping<"largest", "largest", "true">,
    AttributeMapping<"sorted", "sorted", "true">,
    AttributeMapping<"", "index_type", "INT64">
  ];
}


def ONNX_Cast : ONNXExtension<"Cast"> {
  let extension_attr_ = [ ExtensionAttr<"to", Integer, "0"> ];
}

def ONNX_Unsqueeze : ONNXExtension<"Unsqueeze"> {
  let extension_attr_ = [ ExtensionAttr<"axes", IntegerList, "{}"> ];
}

def ONNX_Flatten : ONNXExtension<"Flatten"> { 
  let extension_attr_ = [
    ExtensionAttr<"axis", Integer, "1">,
  ];
}

def ONNX_ConstantOfShape : ONNXExtension<"ConstantOfShape"> { }
def ONNX_Expand : ONNXExtension<"Expand"> { }
def ONNX_Tile : ONNXExtension<"Tile">{}
def ONNX_NonZero : ONNXExtension<"NonZero"> { }

def ONNX_Resize : OpMapping<"Resize", Resize> {
  let attr_mapping_ = [
    AttributeMapping<"", "axes_mask", "-1">,
    AttributeMapping<"", "explicit_shape", "false">
  ];
  let extension_attr_ = [
    ExtensionAttr<"mode_name", String, "\"nearest\"">,
    ];
}

def ONNX_Upsample : OpMapping<"Upsample", Resize> {
  let attr_mapping_ = [
    AttributeMapping<"", "axes_mask", "-1">,
    AttributeMapping<"", "explicit_shape", "false">
  ];
}

def ONNX_ConvTranspose : OpMapping<"ConvTranspose", Conv2DTranspose> {
  let attr_mapping_ = [
    AttributeMapping<"auto_pad", "padding", "EXPLICIT">,
    AttributeMapping<"dilations", "dilations", "{1,1}", -1, 1>,
    AttributeMapping<"strides", "strides", "{1,1}", -1, 1>,
    AttributeMapping<"", "data_format", "NCHW">,
    AttributeMapping<"", "filter_format", "CNHW">,
    AttributeMapping<"group", "group", "1">,
    AttributeMapping<"pads", "padding_left", "{0,0,0,0}", 0>,
    AttributeMapping<"pads", "padding_top", "{0,0,0,0}", 1>,
    AttributeMapping<"pads", "padding_right", "{0,0,0,0}", 2>,
    AttributeMapping<"pads", "padding_bottom", "{0,0,0,0}", 3>];
  let extension_attr_ = [
    ExtensionAttr<"kernel_shape", IntegerList, "{}">];
}

def ONNX_Sigmoid : OpMapping<"Sigmoid", Sigmoid> {
}

def ONNX_LeakyRelu: OpMapping<"LeakyRelu", LeakyRelu> {
  let attr_mapping_ = [
    AttributeMapping<"alpha", "alpha", "0.2">
  ];
}

def ONNX_Pad : ONNXExtension<"Pad"> {
  let extension_attr_ = [
    ExtensionAttr<"pads", IntegerList, "{}">,
    ExtensionAttr<"value", Float, "0">,
    ExtensionAttr<"mode", String, "\"CONSTANT\"">,
    ];
}

// onnx slice op operands order is different from halo ir
def ONNX_Slice : ONNXExtension<"Slice">;

def ONNX_Loop : ONNXExtension<"Loop"> {
  let num_outputs_= 2;
}

def ONNX_Sum : ONNXExtension<"Sum">;
def ONNX_Split :ONNXExtension<"Split"> {
  let extension_attr_ = [
    ExtensionAttr<"split", IntegerList, "{}">,
    ExtensionAttr<"axis", Integer, "0">,
    ];
}

def ONNX_Dropout : ONNXExtension<"Dropout">;
