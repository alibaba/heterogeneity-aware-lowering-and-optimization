//===- constant.h -----------------------------------------------*- C++ -*-===//
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

#ifndef HALO_LIB_IR_CONSTANT_H_
#define HALO_LIB_IR_CONSTANT_H_

#include "halo/lib/framework/data_layout.h"
#include "halo/lib/ir/values.h"

namespace halo {

class Function;
class Module;

/// This class represents constant data, which belongs to function or module.
/// The storage type of T needs to be a trivial type.
class Constant : public IRObject {
 public:
  /// Create a constant object. The data specified by `data_ptr` will be copied
  /// into the object based on the data layout and type.
  explicit Constant(GlobalContext& context, const std::string& name,
                    const Type& type, const DataLayout& data_layout,
                    const void* data_ptr, bool do_splat = false);

  explicit Constant(GlobalContext& context, const std::string& name,
                    const Type& type, const std::vector<std::string>& strings);

  /// Returns the parent object that could be a Module or a Function.
  IRObject* GetParent() const noexcept { return parent_; }

  Constant* Clone(const Constant& from) const;

  void SetData(const Type& type, const void* data_ptr, bool do_splate);

  void SetData(const Type& type, const void* data_ptr) {
    SetData(type, data_ptr, false);
  }

  void SetData(const Type& type, const std::vector<std::string>& strings);

  /// Get the const pointer to the data.
  template <typename T>
  const T* GetDataPtr() const {
    const Type& type = GetResultType(0);
    (void)type;
    HLCHECK(Type::HasNativeType<T>(type));
    return static_cast<const T*>(static_cast<const void*>(data_.data()));
  }

  /// Get the pointer to the data.
  template <typename T>
  T* GetDataPtr() {
    const Type& type = GetResultType();
    (void)type;
    HLCHECK(Type::HasNativeType<T>(type));
    return static_cast<T*>(static_cast<void*>(data_.data()));
  }

  const void* GetRawDataPtr() const { return data_.data(); }

  void* GetRawDataPtr() { return data_.data(); }

  size_t GetElementSizeInBytes() const noexcept {
    return data_layout_.Bytes(GetResultType().GetDataType());
  }

  /// Get the element of data.
  template <typename T>
  const T& GetData(size_t idx) const {
    constexpr bool is_str = std::is_same<T, std::string>();
    if constexpr (is_str) {
      HLCHECK(GetResultType().GetDataType() == DataType::STRING);
      return string_data_.at(idx);
    }
    return GetDataPtr<T>()[idx];
  }

  /// Get the element of data.
  template <typename T>
  T& GetData(size_t idx) {
    constexpr bool is_str = std::is_same<T, std::string>();
    if constexpr (is_str) {
      HLCHECK(GetResultType().GetDataType() == DataType::STRING);
      return string_data_.at(idx);
    }
    return GetDataPtr<T>()[idx];
  }

  /// Get the element of data.
  template <typename T>
  const T& operator[](size_t idx) const {
    return GetDataPtr<T>()[idx];
  }

  /// Get the element of data.
  template <typename T>
  T& operator[](size_t idx) {
    return GetDataPtr<T>()[idx];
  }

  int64_t GetDataAsInt64(size_t idx) const;
  std::vector<int64_t> GetDataAsInt64() const;

  float GetDataAsFloat32(size_t idx) const;

  Kind GetKind() const noexcept override { return Kind::Constant; }

  static inline bool Classof(const Constant* c) { return true; }
  static inline bool Classof(const IRObject* obj) {
    return obj->GetKind() == Kind::Constant;
  }

  /// Print the constant info.
  void Print(std::ostream& os) const override;
  void PrintData(std::ostream* os, size_t num_to_print,
                 bool human_friendly) const;
  /// Check Special constant
  bool IsScalarZero() const;
  bool IsScalarOne() const;
  bool HasSameValueOf(float x) const;

 private:
  Constant(const Constant&);

  IRObject* parent_ = nullptr;
  const DataLayout& data_layout_;
  std::vector<unsigned char> data_;
  std::vector<std::string> string_data_;
  friend class ConstantBuilder;
  friend std::unique_ptr<Constant> std::make_unique<Constant>(const Constant&);
};

} // namespace halo

#endif // HALO_LIB_IR_CONSTANT_H_