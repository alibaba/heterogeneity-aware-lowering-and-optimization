//===- arg_types.td ------------------------------------------*- tblgen -*-===//
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

/// Supported value types used as operand types.
/// Also use to generate DataType enums.
// string
def Str : ValueType<0> { let is_string_ = 1; }

// signed integer
let is_integer_ = 1 in {
  def I1 : ValueType<1>;
  def I8 : ValueType<8>;
  def I16 : ValueType<16>;
  def I32 : ValueType<32>;
  def I64 : ValueType<64>;
}

// unsigned integer
let is_integer_ = 1,
    is_unsigned_ = 1 in {
  def UI8 : ValueType<8>;
  def UI16 : ValueType<16>;
  def UI32 : ValueType<32>;
  def UI64 : ValueType<64>;
}

// float point
def BF16 : ValueType<16> { let alt_name = "BFLOAT16"; }
def F16 : ValueType<16>;
def F32 : ValueType<32>;
def F64:  ValueType<64>;

// quantized
let is_integer_ = 1,
    is_quantized_ = 1 in {
  def QI2 : ValueType<2>;
  def QI3 : ValueType<3>;
  def QI4 : ValueType<4>;
}