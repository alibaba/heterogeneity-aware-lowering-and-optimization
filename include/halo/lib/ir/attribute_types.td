//===- attribute_types.td ------------------------------------*- tblgen -*-===//
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

#ifdef VALUE_TYPE
#else
include "value_type.td"
#endif

/// Common Attribute types
let is_integer_ = 1 in {
  def Bool : ValueType<1, "bool">;
  def Integer : ValueType<32, "int">;
  def BoolList : ValueType<1, "std::vector<bool>"> { let is_array_ = 1; }
  def IntegerList : ValueType<32, "std::vector<int>"> { let is_array_ = 1; }
}

def Float : ValueType<32, "float">;
def FloatList : ValueType<32, "std::vector<float>"> { let is_array_ = 1; }

def String : ValueType<0, "std::string"> {
  let is_string_ = 1;
}

def FunctionPtr : ValueType<0, "halo::Function*"> {
  let is_pointer_ = 1;
}

def BasicBlockPtr: ValueType<0, "halo::BasicBlock*"> {
  let is_pointer_ = 1;
}

/// Enum type defined in base.h
class EnumValueType<string name, list<string> enums = []>
  : ValueType<32, name> {
  list<string> valid_enums_ = enums;
}

def EnumDataType : EnumValueType<"DataType">;
def EnumDataFormat : EnumValueType<"DataFormat",
                                   ["NHWC",
                                    "NCHW",
                                    "NDHWC",
                                    "NCDHW",
                                    "HWCN",
                                    "CNHW",
                                    "DNCHW"]>;
def EnumPadding : EnumValueType<"Padding",
                                ["VALID",
                                 "SAME",
                                 "SAME_LOWER",
                                 "EXPLICIT"]>;
def EnumPadMode : EnumValueType<"PadMode",
                                ["CONSTANT",
                                 "REFLECT",
                                 "SYMMETRIC"]>;
def EnumCodeType : EnumValueType<"CodeType",
                                 ["CORNER",
                                  "CENTER_SIZE",
                                  "CORNER_SIZE"]>;
def EnumPred : EnumValueType<"KindPredicate", ["EQ", "NE",
                                               "GT", "LT",
                                               "GE", "LE"]>;
def EnumResizeMode : EnumValueType<"ResizeMode",
                                  ["HALF_PIXEL",
                                   "HALF_PIXEL_TF",
                                   "ALIGN_CORNERS",
                                   "ASYMMETRIC"
                                  ]>;
def EnumInterpolation: EnumValueType<"Interpolation",
                                    ["NEAREST", "LINEAR", "CUBIC"]>;