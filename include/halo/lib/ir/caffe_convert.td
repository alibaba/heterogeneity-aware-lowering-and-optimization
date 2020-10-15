//===- caffe_convert.td --------------------------------------*- tblgen -*-===//
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

def CAFFE_Eltwise : CAFFEExtension<"Eltwise"> {
  let param_name_ = "eltwise_param";
  let optional_attrs_ = ["operation"];
  let extension_attr_ = [ExtensionAttr<"operation", Integer, "1">,
                         ExtensionAttr<"coeff", FloatList, "{}">];
}

def CAFFE_Convolution : CAFFEExtension<"Convolution"> {
  let param_name_ = "convolution_param";
  let optional_attrs_ = [
    "pad_h", "pad_w", "stride_h", "stride_w", "bias_term", "kernel_h",
    "kernel_w", "group", "num_output"
  ];
  let extension_attr_ = [
    ExtensionAttr<"pad", IntegerList, "{0}">,
    ExtensionAttr<"pad_h", Integer, "0">,
    ExtensionAttr<"pad_w", Integer, "0">,
    ExtensionAttr<"stride", IntegerList, "{1}">,
    ExtensionAttr<"stride_h", Integer, "0">,
    ExtensionAttr<"stride_w", Integer, "0">,
    ExtensionAttr<"dilation", IntegerList, "{1}">,
    ExtensionAttr<"kernel_size", IntegerList, "{}">,
    ExtensionAttr<"kernel_h", Integer, "0">,
    ExtensionAttr<"kernel_w", Integer, "0">,
    ExtensionAttr<"num_output", Integer, "0">,
    ExtensionAttr<"group", Integer, "1">,
    ExtensionAttr<"bias_term", Bool, "true">
  ];
}

def CAFFE_Deconvolution : CAFFEExtension<"Deconvolution"> {
  let param_name_ = "convolution_param";
  let optional_attrs_ = [
    "pad_h", "pad_w", "stride_h", "stride_w", "bias_term", "kernel_h",
    "kernel_w", "group", "num_output"
  ];
  let extension_attr_ = [
    ExtensionAttr<"pad", IntegerList, "{0}">,
    ExtensionAttr<"pad_h", Integer, "0">,
    ExtensionAttr<"pad_w", Integer, "0">,
    ExtensionAttr<"stride", IntegerList, "{1}">,
    ExtensionAttr<"stride_h", Integer, "0">,
    ExtensionAttr<"stride_w", Integer, "0">,
    ExtensionAttr<"dilation", IntegerList, "{1}">,
    ExtensionAttr<"kernel_size", IntegerList, "{}">,
    ExtensionAttr<"kernel_h", Integer, "0">,
    ExtensionAttr<"kernel_w", Integer, "0">,
    ExtensionAttr<"num_output", Integer, "0">,
    ExtensionAttr<"group", Integer, "1">,
    ExtensionAttr<"bias_term", Bool, "true">
  ];
}

def CAFFE_Pooling: CAFFEExtension<"Pooling"> {
  let param_name_ = "pooling_param";
  let optional_attrs_ = [
    "pad_h", "pad_w", "stride_h", "stride_w", "kernel_h", "kernel_w",
    "global_pooling", "round_mode"
  ];
  let extension_attr_ = [
    ExtensionAttr<"pool", Integer, "0">,
    ExtensionAttr<"pad", Integer, "0">,
    ExtensionAttr<"pad_h", Integer, "0">,
    ExtensionAttr<"pad_w", Integer, "0">,
    ExtensionAttr<"stride", Integer, "1">,
    ExtensionAttr<"stride_h", Integer, "1">,
    ExtensionAttr<"stride_w", Integer, "1">,
    ExtensionAttr<"kernel_size", Integer, "0">,
    ExtensionAttr<"kernel_h", Integer, "0">,
    ExtensionAttr<"kernel_w", Integer, "0">,
    ExtensionAttr<"round_mode", Integer, "0">,
    ExtensionAttr<"global_pooling", Bool, "false">
  ];
}

def CAFFE_Reshape : CAFFEExtension<"Reshape"> {
  let param_name_ = "reshape_param";
  let optional_attrs_ = ["shape", "axis", "num_axes"];
  let extension_attr_ = [ExtensionAttr<"shape", IntegerList, "{}">,
                         ExtensionAttr<"axis", Integer, "0">,
                         ExtensionAttr<"num_axes", Integer, "-1">];
}

def CAFFE_Scale : CAFFEExtension<"Scale"> {
  let param_name_ = "scale_param";
  let optional_attrs_ = ["bias_term", "axis", "num_axes"];
  let extension_attr_ = [ExtensionAttr<"bias_term", Bool, "false">,
                         ExtensionAttr<"axis", Integer, "1">,
                         ExtensionAttr<"num_axes", Integer, "1">];
}

def CAFFE_Concat : OpMapping<"Concat", Concat> {
  let param_name_ = "concat_param";
  let optional_attrs_ = ["axis"];
  let attr_mapping_ = [AttributeMapping<"axis", "axis", "1">,
                       AttributeMapping<"", "N", "0">];
}

def CAFFE_Relu : OpMapping<"ReLU", Relu>;

def CAFFE_Softmax : OpMapping<"Softmax", Softmax> {
  let param_name_ = "softmax_param";
  let attr_mapping_ = [AttributeMapping<"axis", "axis", "1">];
}

def CAFFE_LRN : OpMapping<"LRN", LRN> {
  let param_name_ = "lrn_param";
  let attr_mapping_ = [
    AttributeMapping<"local_size", "size", "5">,
    AttributeMapping<"alpha", "alpha", "1">,
    AttributeMapping<"beta", "beta", "5">,
    AttributeMapping<"", "bias", "1.0">,
    AttributeMapping<"", "data_format", "NCHW">
  ];
}

def CAFFE_BatchNorm : OpMapping<"BatchNorm", BatchNorm> {
  let param_name_ = "batch_norm_param";
  let attr_mapping_ = [
    AttributeMapping<"eps", "epsilon", "0.00001">,
    AttributeMapping<"", "data_format", "NCHW">
  ];
}

def CAFFE_Power : CAFFEExtension<"Power"> {
  let param_name_ = "power_param";
  let optional_attrs_ = ["power", "scale", "shift"];
  let extension_attr_ = [ExtensionAttr<"power", Float, "1.0">,
                         ExtensionAttr<"scale", Float, "1.0">,
                         ExtensionAttr<"shift", Float, "0.0">];
}

def CAFFE_InnerProduct: CAFFEExtension<"InnerProduct"> {
  let param_name_ = "inner_product_param";
  let optional_attrs_ = ["num_output", "bias_term", "axis", "transpose"];
  let extension_attr_ = [ExtensionAttr<"num_output", Integer, "1.0">,
                         ExtensionAttr<"bias_term", Bool, "true">,
                         ExtensionAttr<"axis", Integer, "1">,
                         ExtensionAttr<"transpose", Bool, "false">];
}

def CAFFE_Dropout : CAFFEExtension<"Dropout">;

def CAFFE_Upsample : CAFFEExtension<"Upsample"> {
  let param_name_ = "upsample_param";
  let extension_attr_ = [
    ExtensionAttr<"scale", Integer, "1">
  ];
}

def CAFFE_Permute: OpMapping<"Permute", Transpose> {
  let param_name_ = "permute_param";
  let attr_mapping_ = [
    AttributeMapping<"order", "permutation", "{0, 0}">
  ];
}

def CAFFE_PRelu: OpMapping<"PReLU", PRelu> { }