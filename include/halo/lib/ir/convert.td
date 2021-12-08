//===- convert.td --------------------------------------------*- tblgen -*-===//
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

include "instructions.td"

/// extern_attr_name: extern op attribute name string.
/// sn_attr: Halo instruction attribute name string.
/// attr_value: default value used to set Halo attribute.
/// index: index to access the extern attribute vector and map to the Halo
///        scalar attribute; default -1 means type match.
/// expand_dims: flag to indicate post processing attribute value. Apply to
///              spatial only attributes to match [N, C, H0, H1..Hn]
class AttributeMapping<string extern_attr_name,
                       string sn_attr,
                       string attr_value,
                       int index = -1,
                       bit expand_dims = 0> {
  string extern_attr_name_ = extern_attr_name;
  string sn_attr_ = sn_attr;
  string attr_value_ = attr_value;
  int index_ = index;
  bit expand_dims_ = expand_dims;
}

class ExtensionAttr<string extern_attr_name,
                    ValueType value_type,
                    string default_value,
                    bit expand_dims = 0> {
  string extern_attr_name_ = extern_attr_name;
  ValueType value_type_ = value_type;
  string default_value_ = default_value;
  bit expand_dims_ = expand_dims;
}

class OpMapping<string extern_op, Inst sn_inst, code custom_code = ""> {
  string extern_op_ = extern_op;
  Inst sn_inst_ = sn_inst;
  list<AttributeMapping> attr_mapping_ = [];
  list<ExtensionAttr> extension_attr_ = [];
  code custom_code_ = custom_code;
  int num_outputs_ = 1; // Negative for variadic
  // caffe use only
  string param_name_ = "";
  list<string> optional_attrs_ = [];
  list<int> optional_args_ = [];
}

// A marker to represent ExtensionInst
def Extension : Inst<"Framework extension instruction"> {
  let cat_ = cat_common;
}

/// Class to map a framwork op to Halo ExtensionInst.
class OpExtension<string extern_op> : OpMapping<extern_op, Extension>;

// Subclass to map Tensorflow op.
class TFExtension<string extern_op> : OpExtension<extern_op>;

// Subclass to map Caffe op.
class CAFFEExtension<string extern_op> : OpExtension<extern_op>;

// Subclass to map ONNX op.
class ONNXExtension<string extern_op> : OpExtension<extern_op>;

// Subclass to map TFLITE op.
class TFLITEExtension<string extern_op> : OpExtension<extern_op>;

// Subclass to map MXNET op.
class MXNETExtension<string extern_op> : OpExtension<extern_op>;
