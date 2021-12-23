//===- value_type.td -----------------------------------------*- tblgen -*-===//
//
// Copyright (C) 2019-2021 Alibaba Group Holding Limited.
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
#define VALUE_TYPE

/// Base class for Type.
class ValueType<int size, string cpp_ty = ""> {
  // bit width
  int width_ = size;
  // mapped cpp type string
  string cpp_type_ = cpp_ty;
  bit is_array_ = 0;
  // integer type flag
  bit is_integer_ = 0;
  // string type flag
  bit is_string_ = 0;
  // pointer type flag
  bit is_pointer_ = 0;
  // unsigned type flag
  bit is_unsigned_ = 0;
  // quantized type flag
  bit is_quantized_ = 0;
  // 2d array type flag
  bit is_2d_array_ = 0;
  // alternative name
  string alt_name = "";
}

#endif // VALUE_TYPE
