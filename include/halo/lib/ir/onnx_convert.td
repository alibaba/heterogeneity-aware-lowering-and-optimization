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

def ONNX_Abs : OpMapping<"Abs", Abs>;
def ONNX_Acos : OpMapping<"Acos", ACos>;
def ONNX_Acosh : OpMapping<"Acosh", ACosh>;
def ONNX_Add : OpMapping<"Add", Add>;
def ONNX_And : OpMapping<"And", And>;

def ONNX_Sign: OpMapping<"Sign",Sign>;

def ONNX_IsInf: OpMapping<"IsInf", IsInf>{
  let attr_mapping_ = [
    AttributeMapping<"detect_negative", "detect_negative", "true">,
    AttributeMapping<"detect_positive", "detect_positive", "true">
  ];
}

def ONNX_IsNaN: OpMapping<"IsNaN", IsNaN>;

def ONNX_ArgMax : OpMapping<"ArgMax", Argmax> {
 let attr_mapping_ = [
   AttributeMapping<"axis", "axis", "0">,
   AttributeMapping<"", "keep_dims", "false">,
   AttributeMapping<"select_last_index", "select_last_index", "false">
 ];
}

def ONNX_ArgMin : OpMapping<"ArgMin", Argmin> {
 let attr_mapping_ = [
   AttributeMapping<"axis", "axis", "0">,
   AttributeMapping<"", "keep_dims", "false">,
   AttributeMapping<"select_last_index", "select_last_index", "false">
 ];
}

def ONNX_Asin : OpMapping<"Asin", ASin>;
def ONNX_Asinh : OpMapping<"Asinh", ASinh>;
def ONNX_Atan : OpMapping<"Atan", ATan>;
def ONNX_Atanh : OpMapping<"Atanh", ATanh>;

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
    AttributeMapping<"ceil_mode", "round_mode", "0">,
    AttributeMapping<"group", "group", "1">];
  let extension_attr_ = [
    ExtensionAttr<"dilations", IntegerList, "{1,1}">,
    ExtensionAttr<"storage_order", Integer, "0">];
}

let attr_mapping_ = [
    AttributeMapping<"epsilon", "epsilon", "0.00001">,
    AttributeMapping<"", "data_format", "NCHW">] in {
  def ONNX_BatchNormalization : OpMapping<"BatchNormalization", BatchNorm>;
  def ONNX_InstanceNormalization : OpMapping<"InstanceNormalization", InstanceNorm>;
}

def ONNX_Eyelike : ONNXExtension<"EyeLike"> {
  let extension_attr_ = [
    ExtensionAttr<"dtype", Integer, "-1">,
    ExtensionAttr<"k", Integer, "0"> ];
}

def ONNX_GroupNormalization : OpMapping<"GroupNormalization", GroupNorm> {
  let attr_mapping_ = [
    AttributeMapping<"epsilon", "epsilon", "0.00001">,
    AttributeMapping<"num_groups", "groups", "-1">,
    AttributeMapping<"", "data_format", "NCHW">];
}

def ONNX_BitShift : OpMapping<"BitShift", Shift,
    [{ inst->SetIsLeftShift(direction == "LEFT"); }]> {
  let extension_attr_ = [ExtensionAttr<"direction", String, "\"\"">];
}

def ONNX_Cast : ONNXExtension<"Cast"> {
  let extension_attr_ = [ ExtensionAttr<"to", Integer, "0"> ];
}

def ONNX_Compress : OpMapping<"Compress", Compress> {
  let attr_mapping_ = [ AttributeMapping<"axis", "axis", "std::numeric_limits<int>::max()"> ];
}

def ONNX_IpuAttentionMask : ONNXExtension<"AttentionMask">;

def ONNX_IpuGelu : ONNXExtension<"Gelu">;

def ONNX_Clip : ONNXExtension<"Clip"> {
  let extension_attr_ = [
    ExtensionAttr<"max", Float, "std::numeric_limits<float>::max()">,
    ExtensionAttr<"min", Float, "std::numeric_limits<float>::lowest()">
  ];
}

def ONNX_Ceil : OpMapping<"Ceil", Ceil>;

def ONNX_Celu : OpMapping<"Celu", Celu> {
  let attr_mapping_ = [ AttributeMapping<"alpha", "alpha", "1.0"> ];
}

def ONNX_Round : OpMapping<"Round", Round>;

// def ONNX_Compress
def ONNX_Concat : OpMapping<"Concat", Concat> {
  let attr_mapping_ = [AttributeMapping<"axis", "axis", "1">,
                       AttributeMapping<"", "N", "0">];
}
// def ONNX_Constant
def ONNX_ConstantOfShape : ONNXExtension<"ConstantOfShape">;

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

def ONNX_Cos: OpMapping<"Cos", Cos>;

def ONNX_Cosh : OpMapping<"Cosh", Cosh>;

def ONNX_CumSum : OpMapping<"CumSum", CumSum> {
  let attr_mapping_ = [
    AttributeMapping<"exclusive", "exclusive", "false">,
    AttributeMapping<"reverse", "reverse", "false">
  ];
}

def ONNX_DepthToSpace : ONNXExtension<"DepthToSpace"> {
  let extension_attr_ = [
    ExtensionAttr<"blocksize", Integer, "-1">,
    ExtensionAttr<"mode", String, "\"DCR\"">
  ];
}

def ONNX_DequantizeLinear: OpMapping<"DequantizeLinear", Dequantize> {
   let attr_mapping_ = [ AttributeMapping<"axis", "axis", "1"> ];
}

def ONNX_Det: OpMapping<"Det", Det>;

def ONNX_DynamicQuantizeLinear: ONNXExtension<"DynamicQuantizeLinear">;

def ONNX_Div : OpMapping<"Div", Div>;

def ONNX_Dropout : ONNXExtension<"Dropout">;

def ONNX_Elu : OpMapping<"Elu", Elu> {
  let attr_mapping_ = [ AttributeMapping<"alpha", "alpha", "1.0"> ];
}

def ONNX_Erf : OpMapping<"Erf", Erf>;

def ONNX_Equal: OpMapping<"Equal", Cmp> {
  let attr_mapping_ = [AttributeMapping<"", "predicator", "EQ">];
}

def ONNX_Einsum: OpMapping<"Einsum", Einsum> {
  let attr_mapping_ = [AttributeMapping<"equation", "equation", "\"\"">];
}

def ONNX_Exp : OpMapping<"Exp", Exp>;

def ONNX_Expand : OpMapping<"Expand", ExpandDims>;

def ONNX_Flatten : ONNXExtension<"Flatten"> {
  let extension_attr_ = [
    ExtensionAttr<"axis", Integer, "1">,
  ];
}

def ONNX_Floor : OpMapping<"Floor", Floor>;

def ONNX_Gather : OpMapping<"Gather", Gather> {
  let attr_mapping_ = [AttributeMapping<"axis", "axis", "0">];
}

def ONNX_GatherElements : OpMapping <"GatherElements", GatherElements> {
  let attr_mapping_ = [AttributeMapping<"axis", "axis", "0">];
}

def ONNX_Gemm: OpMapping<"Gemm", Gemm> {
  let attr_mapping_ = [
    AttributeMapping<"alpha", "alpha", "1.0">,
    AttributeMapping<"beta", "beta", "1.0">,
    AttributeMapping<"transA", "transpose_a", "0">,
    AttributeMapping<"transB", "transpose_b", "0">];
}

// TODO (unknown): GlobalAveragePool op averages data within the same channel.
// For ND image where N>2, axis = range(shape.size()-2, shape.size())
def ONNX_GlobalAveragePool : OpMapping<"GlobalAveragePool", ReduceMean> {
  let attr_mapping_ = [
    AttributeMapping<"", "axis", "{2,3}">,
    AttributeMapping<"", "keep_dims", "true">];
}

def ONNX_GlobalMaxPool : ONNXExtension<"GlobalMaxPool">;

def ONNX_Greater : OpMapping<"Greater", Cmp> {
  let attr_mapping_ = [AttributeMapping<"", "predicator", "GT">];
}

def ONNX_GreaterOrEqual : OpMapping<"GreaterOrEqual", Cmp> {
  let attr_mapping_ = [AttributeMapping<"", "predicator", "GE">];
}

def ONNX_Identity : ONNXExtension<"Identity">;

def ONNX_LeakyRelu : OpMapping<"LeakyRelu", LeakyRelu> {
  let attr_mapping_ = [ AttributeMapping<"alpha", "alpha", "0.01"> ];
}

def ONNX_Less : OpMapping<"Less", Cmp> {
  let attr_mapping_ = [AttributeMapping<"", "predicator", "LT">];
}

def ONNX_LessOrEqual : OpMapping<"LessOrEqual", Cmp> {
  let attr_mapping_ = [AttributeMapping<"", "predicator", "LE">];
}

def ONNX_Log : OpMapping<"Log", Log>;

// def ONNX_Loop : ONNXExtension<"Loop"> {
//  let num_outputs_= 2;
// }

def ONNX_LRN : OpMapping<"LRN", LRN> {
  let attr_mapping_ = [
    AttributeMapping<"size", "size", "0">,
    AttributeMapping<"alpha", "alpha", "0.00001">,
    AttributeMapping<"beta", "beta", "0.75">,
    AttributeMapping<"bias", "bias", "1.0">,
    AttributeMapping<"", "data_format", "NCHW">
  ];
}

def ONNX_MatMul: OpMapping<"MatMul", MatMul> {
  let attr_mapping_ = [
    AttributeMapping<"", "transpose_a", "false">,
    AttributeMapping<"", "transpose_b", "false">
  ];
}

def ONNX_Max : ONNXExtension<"Max">;

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
    AttributeMapping<"group", "group", "1">,
    AttributeMapping<"ceil_mode", "round_mode", "0">];

  let extension_attr_ = [
    ExtensionAttr<"dilations", IntegerList, "{1,1}">,
    ExtensionAttr<"storage_order", Integer, "0">];
}

def ONNX_Mean : OpMapping<"Mean", Mean>;

def ONNX_Min : ONNXExtension<"Min">;

def ONNX_Mod : OpMapping<"Mod", Mod> {
   let attr_mapping_ = [
     AttributeMapping<"fmod", "fmod", "0">,
   ];
 }

def ONNX_Mul : OpMapping<"Mul", Mul>;

def ONNX_Neg : OpMapping<"Neg", Neg>;

def ONNX_NLLL : OpMapping<"NegativeLogLikelihoodLoss", NegativeLogLikelihoodLoss> {
  let attr_mapping_ = [
    AttributeMapping<"ignore_index", "ignored", "-1">,
    AttributeMapping<"reduction", "reduction", "MEAN">,
  ];
}

def ONNX_NonMaxSupression : OpMapping<"NonMaxSuppression", NonMaxSuppression> {
  let attr_mapping_ = [
    AttributeMapping<"", "eta", "1.0">,
    AttributeMapping<"", "score_threshold", "-1.0">,
    AttributeMapping<"", "index_type", "INT64">];
}

def ONNX_NonZero : ONNXExtension<"NonZero">;

def ONNX_Not : OpMapping<"Not", Not>;

def ONNX_OneHot : ONNXExtension<"OneHot"> {
  let extension_attr_ = [ ExtensionAttr<"axis", Integer, "-1"> ];
}

def ONNX_Or : OpMapping<"Or", Or>;

def ONNX_Xor : OpMapping<"Xor", Xor>;

def ONNX_Pad : ONNXExtension<"Pad"> {
  let extension_attr_ = [
    ExtensionAttr<"mode", EnumPadMode, "CONSTANT">,
    ExtensionAttr<"pads", IntegerList, "{}">,
    ExtensionAttr<"values", Float, "0">
  ];
}

def ONNX_Pow : OpMapping<"Pow", Pow>;

def ONNX_PRelu: OpMapping<"PRelu", PRelu>;

def ONNX_QuantizeLinear: OpMapping<"QuantizeLinear", Quantize> {
   let attr_mapping_ = [
     AttributeMapping<"axis", "axis", "1">,
     AttributeMapping<"", "bits", "8">,
     AttributeMapping<"", "sign_bit", "false">
   ];
}

def ONNX_Range:  ONNXExtension<"Range">;

def ONNX_Reciprocal : OpMapping<"Reciprocal", Rcp>;

let attr_mapping_ = [
    AttributeMapping<"keepdims", "keep_dims", "true">,
    AttributeMapping<"axes", "axis", "{}"> ] in {
  def ONNX_ReduceL1 : OpMapping<"ReduceL1", ReduceL1>;
  def ONNX_ReduceL2 : OpMapping<"ReduceL2", ReduceL2>;
  def ONNX_ReduceLogSum : OpMapping<"ReduceLogSum", ReduceLogSum>;
  def ONNX_ReduceLogSumExp : OpMapping<"ReduceLogSumExp", ReduceLogSumExp>;
  def ONNX_ReduceMean: OpMapping<"ReduceMean", ReduceMean>;
  def ONNX_ReduceMax: OpMapping<"ReduceMax", ReduceMax>;
  def ONNX_ReduceMin: OpMapping<"ReduceMin", ReduceMin>;
  def ONNX_ReduceSum : OpMapping<"ReduceSum", ReduceSum>;
  def ONNX_ReduceProd: OpMapping<"ReduceProd", ReduceProduct>;
  def ONNX_ReduceSumSquare : OpMapping<"ReduceSumSquare", ReduceSumSquare>;
}

def ONNX_Relu : OpMapping<"Relu", Relu>;

def ONNX_Reshape : OpMapping<"Reshape", Reshape>;

def ONNX_Resize : ONNXExtension<"Resize">;

def ONNX_Selu : OpMapping<"Selu", Selu> {
  let attr_mapping_ = [
    AttributeMapping<"gamma", "lambda", "1.0507">,
    AttributeMapping<"alpha", "alpha", "1.67326">
    ];
}

def ONNX_Shape : ONNXExtension<"Shape">;

def ONNX_Shrink : OpMapping<"Shrink", Shrink>{
  let attr_mapping_ = [
    AttributeMapping<"bias", "bias", "0.0">,
    AttributeMapping<"lambd", "lambd", "0.5">
    ];
}

def ONNX_Sigmoid : OpMapping<"Sigmoid", Sigmoid>;

def ONNX_HardSigmoid : OpMapping<"HardSigmoid", HardSigmoid>{
  let attr_mapping_ = [
    AttributeMapping<"alpha", "alpha", "0.2">,
    AttributeMapping<"beta", "beta", "0.5">
    ];
}

def ONNX_Sin : OpMapping<"Sin", Sin>;

def ONNX_Sinh : OpMapping<"Sinh", Sinh>;

def ONNX_Size: ONNXExtension<"Size">;

// onnx slice op operands order is different from halo ir
def ONNX_Slice : ONNXExtension<"Slice"> {
  let extension_attr_ = [
    ExtensionAttr<"axes", IntegerList, "{}">,
    ExtensionAttr<"starts", IntegerList, "{}">,
    ExtensionAttr<"ends", IntegerList, "{}">,
  ];
}

let attr_mapping_ = [AttributeMapping<"axis", "axis", "-1">] in {
  def ONNX_Hardmax : OpMapping<"Hardmax", Hardmax>;
  def ONNX_Softmax : OpMapping<"Softmax", Softmax>;
  def ONNX_LogSoftmax : OpMapping<"LogSoftmax", LogSoftmax>;
}

def ONNX_SoftmaxCrossEntropy : ONNXExtension<"SoftmaxCrossEntropyLoss"> {
  let extension_attr_ = [
    ExtensionAttr<"ignore_index", Integer, "-1">,
    ExtensionAttr<"reduction", EnumReduction, "MEAN">,
  ];
}

def ONNX_Softplus : OpMapping<"Softplus", Softplus>;

def ONNX_Softsign : OpMapping<"Softsign", Softsign>;

def ONNX_Split : ONNXExtension<"Split"> {
  let extension_attr_ = [
    ExtensionAttr<"split", IntegerList, "{}">,
    ExtensionAttr<"axis", Integer, "0">,
    ];
  let num_outputs_ = -1; // Variadic
}

def ONNX_Sqrt : OpMapping<"Sqrt", Sqrt>;

def ONNX_Squeeze : ONNXExtension<"Squeeze"> {
  let extension_attr_ = [ ExtensionAttr<"axes", IntegerList, "{}"> ];
}

def ONNX_Sub : OpMapping<"Sub", Sub>;

def ONNX_Sum : ONNXExtension<"Sum">;

def ONNX_TopK : OpMapping<"TopK", TopK> {
  let attr_mapping_ = [
    AttributeMapping<"axis", "axis", "-1">,
    AttributeMapping<"largest", "largest", "true">,
    AttributeMapping<"sorted", "sorted", "true">,
    AttributeMapping<"", "index_type", "INT64">
  ];
}

def ONNX_Tanh : OpMapping<"Tanh", Tanh>;

def ONNX_Tan : OpMapping<"Tan", Tan>;

def ONNX_ThresholdedRelu : OpMapping<"ThresholdedRelu", ThresholdedRelu> {
  let attr_mapping_ = [ AttributeMapping<"alpha", "alpha", "1"> ];
}

def ONNX_Tile : OpMapping<"Tile", Tile>;

def ONNX_Transpose : OpMapping<"Transpose", Transpose> {
  let attr_mapping_ = [
    AttributeMapping<"perm", "permutation", "{}">];
}

def ONNX_Unsqueeze : ONNXExtension<"Unsqueeze"> {
  let extension_attr_ = [ ExtensionAttr<"axes", IntegerList, "{}"> ];
}

def ONNX_Upsample : OpMapping<"Upsample", Resize> {
  let attr_mapping_ = [
    AttributeMapping<"", "axes_mask", "-1">,
    AttributeMapping<"", "explicit_shape", "false">
  ];
}

def ONNX_HgEngine : ONNXExtension<"HgEngine"> {
  let extension_attr_ = [
    ExtensionAttr<"serialized_engine", String, "{}">,
    ExtensionAttr<"in_data_format", String, "{}">,
    ExtensionAttr<"out_data_format", String, "{}">,
    ExtensionAttr<"_output_shapes", String, "{}">,
    ExtensionAttr<"in_binding_list", String, "{}">,
    ExtensionAttr<"out_binding_list", String, "{}">,
    ExtensionAttr<"in_type_list", String, "{}">,
    ExtensionAttr<"out_type_list", String, "{}">,];
}

def ONNX_HgQuant : ONNXExtension<"HgQuant"> {
  let extension_attr_ = [
    ExtensionAttr<"in_scale", String, "\"1.0\"">,
    ExtensionAttr<"in_bias", String, "\"0.0\"">,
    ExtensionAttr<"qtype", String, "\"int8\"">,
    ExtensionAttr<"is_per_channel", Integer, "0">,
    ExtensionAttr<"round_mode", Integer, "0">,
    ExtensionAttr<"in_data_format", String, "\"NHWC\"">,
    ExtensionAttr<"out_data_format", String, "\"NHWC\"">,
    ExtensionAttr<"model_name", String, "\"\"">,
    ExtensionAttr<"op_name", String, "\"\"">,
    ExtensionAttr<"in_type", String, "\"FLOAT32\"">,
    ExtensionAttr<"out_type", String, "\"INT8\"">
  ];
}

def ONNX_HgDequant : ONNXExtension<"HgDequant"> {
  let extension_attr_ = [
    ExtensionAttr<"in_scale", String, "\"1.0\"">,
    ExtensionAttr<"in_bias", String, "\"0.0\"">,
    ExtensionAttr<"is_per_channel", Integer, "0">,
    ExtensionAttr<"in_data_format", String, "\"NHWC\"">,
    ExtensionAttr<"out_data_format", String, "\"NHWC\"">,
    ExtensionAttr<"model_name", String, "\"\"">,
    ExtensionAttr<"op_name", String, "\"\"">,
    ExtensionAttr<"in_type", String, "\"INT8\"">,
    ExtensionAttr<"out_type", String, "\"FLOAT32\"">
  ];
}

defvar RNN_attrs = [
    ExtensionAttr<"hidden_size", Integer, "1">,
    ExtensionAttr<"layout", Integer, "0">,
    ExtensionAttr<"direction", String, "\"FORWARD\"">
];

def ONNX_LSTM : ONNXExtension<"LSTM"> {
  let extension_attr_ = RNN_attrs;
  let optional_args_  = [3, 4, 5, 6, 7];
}

def ONNX_GRU : ONNXExtension<"GRU"> {
  let extension_attr_ = !listconcat(RNN_attrs, [
    ExtensionAttr<"linear_before_reset", Integer, "0">,
  ]);
}

def ONNX_RNN : ONNXExtension<"RNN"> {
  let extension_attr_ = RNN_attrs;
}

def ONNX_TFIDFVec : ONNXExtension<"TfIdfVectorizer"> {
  let extension_attr_ = [
    ExtensionAttr<"min_gram_length", Integer, "1">,
    ExtensionAttr<"max_gram_length", Integer, "1">,
    ExtensionAttr<"max_skip_count", Integer, "1">,
    ExtensionAttr<"mode", EnumTFIDF, "INVALID">,
    ExtensionAttr<"ngram_counts", Integer64List, "{}">,
    ExtensionAttr<"ngram_indexes", Integer64List, "{}">,
    ExtensionAttr<"pool_int64s", Integer64List, "{}">,
    ExtensionAttr<"pool_strings", StringList, "{}">,
    ExtensionAttr<"weights", FloatList, "{}">,
  ];
}

def ONNX_WHERE: OpMapping<"Where", Select>;