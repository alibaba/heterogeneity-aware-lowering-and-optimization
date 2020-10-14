//===- halo_data.h ----------------------------------------------*- C++ -*-===//
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

#ifndef HALO_API_HALO_DATA_H_
#define HALO_API_HALO_DATA_H_

#include <cstdint>

namespace halo {

/// Status IDs.
enum class Status {
  SUCCESS = 0,
  ASSERTION,
  COMPILE_FAILURE,
  FILE_NOT_EXIST,
  ILLEGAL_PARAM,
  INTERPRET_FAILURE,
  NULL_PTR,
};

/// Data Type IDs.
enum class DataType {
#define GET_DATATYPE_ENUM_VALUE
#include "halo/lib/ir/datatype.def"
#undef GET_DATATYPE_ENUM_VALUE
  INVALID,
};

/// Instruction Op Code.
enum class OpCode {
#define GET_INST_INFO_OPCODE_ENUM
#include "halo/lib/ir/instructions_info.def"
#undef GET_INST_INFO_OPCODE_ENUM
  // Custom opcode
  CUSTOM,
  // Framework extension opcode
  EXTENSION,
  // Invalid OpCode
  INVALID,
};

/// TensorFlow Extension Op Code
enum class TFExtOpCode {
#define GET_TF_OPCODE_ENUM
#include "halo/lib/ir/convert_tf_info.def"
#undef GET_TF_OPCODE_ENUM
  UNSUPPORTED,
};

/// ONNX Extension Op Code
enum class ONNXExtOpCode {
#define GET_ONNX_OPCODE_ENUM
#include "halo/lib/ir/convert_onnx_info.def"
#undef GET_ONNX_OPCODE_ENUM
  UNSUPPORTED,
};

/// TFLITE Extension Op Code
enum class TFLITEExtOpCode {
#define GET_TFLITE_OPCODE_ENUM
#include "halo/lib/ir/convert_tflite_info.def"
#undef GET_TFLITE_OPCODE_ENUM
  UNSUPPORTED,
};

/// CAFFE Extension Op Code
enum class CAFFEExtOpCode {
#define GET_CAFFE_OPCODE_ENUM
#include "halo/lib/ir/convert_caffe_info.def"
#undef GET_CAFFE_OPCODE_ENUM
  UNSUPPORTED,
};

/// User defined Op Code
enum class CustomExtOpCode {
  UNSUPPORTED,
};

} // end of namespace halo

#endif // HALO_API_HALO_DATA_H_