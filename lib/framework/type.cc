//===- type.cc ------------------------------------------------------------===//
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

#include "halo/lib/framework/type.h"

#include <iostream>
#include <numeric>

#include "halo/api/halo_data.h"
#include "halo/lib/framework/common.h"
#include "halo/lib/framework/global_context.h"

namespace halo {

/// To constructs a scalar dt_id type.
Type::Type(const DataType dt_id) : data_type_id_(dt_id) {
  is_scalar_ = true;
  total_num_of_elements_ = 1;
}

/// To constructs a dt_id type with the shape dimension.
/// An empty shape is allowed.
Type::Type(const DataType dt_id, const std::vector<int64_t>& shape)
    : data_type_id_(dt_id), shape_(shape) {
  if (!shape.empty()) {
    total_num_of_elements_ = 1;
    for (int64_t dim_size : shape) {
      if (dim_size < 0) {
        if (dim_size == kDynamicBatchSize) {
          total_num_of_elements_ = std::numeric_limits<int64_t>::max();
          is_dynamic_batch_ = true;
        } else {
          total_num_of_elements_ = -1;
        }
        break;
      }
      total_num_of_elements_ *= dim_size;
    }
  } else {
    // scalar type
    is_scalar_ = true;
    total_num_of_elements_ = 1;
  }
}

template <>
bool Type::HasNativeType<bool>(DataType dt) {
  return dt == DataType::BOOL;
}

template <>
bool Type::HasNativeType<int8_t>(DataType dt) {
  return dt == DataType::INT8 || dt == DataType::BOOL;
}

template <>
bool Type::HasNativeType<uint8_t>(DataType dt) {
  return dt == DataType::UINT8;
}

template <>
bool Type::HasNativeType<int16_t>(DataType dt) {
  return dt == DataType::INT16;
}

template <>
bool Type::HasNativeType<uint16_t>(DataType dt) {
  return dt == DataType::UINT32;
}

template <>
bool Type::HasNativeType<int32_t>(DataType dt) {
  return dt == DataType::INT32 || dt == DataType::UINT32;
}

template <>
bool Type::HasNativeType<uint32_t>(DataType dt) {
  return dt == DataType::UINT32 || dt == DataType::INT32;
}

template <>
bool Type::HasNativeType<float>(DataType dt) {
  return dt == DataType::FLOAT32;
}

template <>
bool Type::HasNativeType<int64_t>(DataType dt) {
  return dt == DataType::INT64 || dt == DataType::UINT64;
}

template <>
bool Type::HasNativeType<uint64_t>(DataType dt) {
  return dt == DataType::INT64 || dt == DataType::UINT64;
}

std::string Type::DataTypeToString(DataType dt) {
  std::string s;
  switch (dt) {
#define GET_DATATYPE_ENUM_STRING
#include "halo/lib/ir/datatype.def"
#undef GET_DATATYPE_ENUM_STRING
    default:
      s = "invalid";
  }
  return s;
}

/// Print out the type info.
void Type::Print(std::ostream& os) const {
  os << "[";
  os << DataTypeToString(data_type_id_);
  os << ": ";
  int idx = 0;
  for (auto d : shape_) {
    if (idx++ > 0) {
      os << "x";
    }
    os << d;
  }
  os << "]";
}

bool Type::operator==(const Type& rhs) const noexcept {
  if (!IsValid() || !rhs.IsValid()) {
    return false;
  }
  if (is_scalar_ ^ rhs.IsScalar()) {
    return false;
  }
  if (data_type_id_ != rhs.GetDataType() ||
      shape_.size() != rhs.GetNumOfDims()) {
    return false;
  }
  for (int i = 0, e = GetNumOfDims(); i < e; ++i) {
    if (shape_[i] != rhs.GetNumOfElementsInDim(i)) {
      return false;
    }
  }
  return true;
}

bool Type::operator!=(const Type& rhs) const noexcept {
  return !this->operator==(rhs);
}

void Type::Dump() const { Print(GlobalContext::Dbgs()); }

} // namespace halo