//===- type.h ---------------------------------------------------*- C++ -*-===//
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

#ifndef HALO_LIB_FRAMEWORK_TYPE_H_
#define HALO_LIB_FRAMEWORK_TYPE_H_

#include <cstdint>
#include <iostream>
#include <vector>

#include "halo/api/halo_data.h"
#include "halo/lib/framework/common.h"

namespace halo {

const int kDynamicBatchSize = -1;

/// This class defines the type for a value in the IR system.
/// It includes a data type ID, and the dimension sizes.
class Type final {
 public:
  /// The default constructor creates a scalar type which is
  /// DataType::INVALID.
  Type() : Type(DataType::INVALID) {}

  /// To constructs a scalar dt_id type.
  explicit Type(DataType dt_id);

  /// To constructs a dt_id type with the shape.
  explicit Type(DataType dt_id, const std::vector<int64_t>& shape);

  Type(const Type& src_type) = default;
  Type(Type&& src_type) = default;
  Type& operator=(const Type& src_type) = default;
  Type& operator=(Type&& src_type) = default;
  bool operator==(const Type& rhs) const noexcept;
  bool operator!=(const Type& rhs) const noexcept;

  ~Type() = default;

  /// Return the total number of elements.
  int64_t GetTotalNumOfElements() const noexcept {
    return total_num_of_elements_;
  }

  /// Return the number of dimensions.
  size_t GetNumOfDims() const noexcept { return shape_.size(); }

  /// Return the number of elements in one dim.
  int64_t GetNumOfElementsInDim(size_t d) const {
    HLCHECK(d < shape_.size());
    return shape_[d];
  }

  const std::vector<int64_t>& GetDimSizes() const noexcept { return shape_; }

  DataType GetDataType() const noexcept { return data_type_id_; }

  /// Return true if it is a scalar type.
  bool IsScalar() const noexcept { return is_scalar_; }

  /// Return true if it is a scalar type.
  bool IsDynamicBatch() const noexcept { return is_dynamic_batch_; }

  /// Returns true if it has a valid type.
  bool IsValid() const noexcept {
    return (data_type_id_ != DataType::INVALID && total_num_of_elements_ >= 0);
  }

  /// Returns true if this type can be broadcasted to `to`.
  bool BroadcastableTo(const Type& to) const noexcept;

  /// Return the number of non-one dimensions.
  size_t GetSqueezedNumOfDims() const noexcept;

  /// Print out the info.
  void Print(std::ostream& os) const;

  /// Dump the info.
  void Dump() const;

  /// Checks if the DataType can be mapped to native c/c++ type.
  template <typename T>
  static bool HasNativeType(DataType dt) {
    return false;
  }

  template <typename T>
  static bool HasNativeType(const Type& type) {
    return HasNativeType<T>(type.GetDataType());
  }

  static bool IsIntegerType(const DataType& dt) {
    return (dt == DataType::BOOL || dt == DataType::INT8 ||
            dt == DataType::UINT8 || dt == DataType::INT16 ||
            dt == DataType::UINT16 || dt == DataType::INT32 ||
            dt == DataType::UINT32 || dt == DataType::INT64 ||
            dt == DataType::UINT64);
  }

  static bool IsIntegerType(const Type& type) {
    return IsIntegerType(type.GetDataType());
  }

  static bool IsFloatingPointType(const DataType& dt) {
    return (dt == DataType::FLOAT16 || dt == DataType::FLOAT32 ||
            dt == DataType::FLOAT64 || dt == DataType::BFLOAT16);
  }

  static bool IsFloatingPointType(const Type& type) {
    return IsFloatingPointType(type.GetDataType());
  }

  static std::string DataTypeToString(DataType dt);
  static DataType StringToDataType(const std::string& name);

 private:
  /// The ID of this type.
  DataType data_type_id_ = DataType::INVALID;

  /// Scalar flag, the default is false.
  bool is_scalar_ = false;

  /// Dynamic batch flag, the default is false.
  bool is_dynamic_batch_ = false;

  /// The total number of elements.
  int64_t total_num_of_elements_ = -1;

  /// Shape indicates the number of elements in each dimension.
  ///
  /// The shape of a scalar type is empty, and the total number
  /// of elements is one.
  std::vector<int64_t> shape_;
};

template <>
bool Type::HasNativeType<bool>(DataType dt);
template <>
bool Type::HasNativeType<int8_t>(DataType dt);
template <>
bool Type::HasNativeType<uint8_t>(DataType dt);
template <>
bool Type::HasNativeType<int16_t>(DataType dt);
template <>
bool Type::HasNativeType<uint16_t>(DataType dt);
template <>
bool Type::HasNativeType<int32_t>(DataType dt);
template <>
bool Type::HasNativeType<uint32_t>(DataType dt);
template <>
bool Type::HasNativeType<float>(DataType dt);
template <>
bool Type::HasNativeType<double>(DataType dt);
template <>
bool Type::HasNativeType<int64_t>(DataType dt);
template <>
bool Type::HasNativeType<uint64_t>(DataType dt);

} // namespace halo

#endif // HALO_LIB_FRAMEWORK_TYPE_H_