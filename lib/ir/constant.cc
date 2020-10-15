//===- constant.cc --------------------------------------------------------===//
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

#include "halo/lib/ir/constant.h"

#include <algorithm>
#include <iostream>

#include "halo/lib/ir/function.h"
#include "halo/lib/ir/module.h"

namespace halo {

Constant::Constant(GlobalContext& context, const std::string& name,
                   const Type& ty, const DataLayout& data_layout,
                   const void* data_ptr, bool do_splat)
    : IRObject(context, name, 1), parent_(nullptr), data_layout_(data_layout) {
  HLCHECK(ty.IsValid());
  SetData(ty, data_ptr, do_splat);
}

Constant::Constant(const Constant& from)
    : IRObject(from.GetGlobalContext(), from.GetName(), 1),
      parent_(nullptr),
      data_layout_(from.data_layout_),
      data_(from.data_) {}

template <typename T>
static void PrintValues(std::ostream* os, const T* ptr, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    if (i > 0) {
      *os << ", ";
    }
    *os << ptr[i]; // NOLINT.
  }
}

void Constant::SetData(const Type& ty, const void* data_ptr, bool do_splat) {
  auto& results = GetResultsTypes();
  results.resize(1);
  results[0] = ty;
  size_t bytes = data_layout_.Bytes(ty);
  data_.resize(bytes);
  const unsigned char* src = static_cast<const unsigned char*>(data_ptr);
  if (!do_splat) {
    std::copy_n(src, bytes, data_.data());
  } else {
    size_t byte_per_element = data_layout_.Bytes(ty.GetDataType());
    for (size_t i = 0, e = bytes / byte_per_element; i != e;
         i += byte_per_element) {
      std::copy_n(src, byte_per_element, data_.begin() + i);
    }
  }
}

// print invisible characters, whitespaces, etc
template <>
void PrintValues<uint8_t>(std::ostream* os, const uint8_t* ptr, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    if (i > 0) {
      *os << ", ";
    }
    *os << static_cast<int32_t>(ptr[i]); // NOLINT.
  }
}

template <>
void PrintValues<int8_t>(std::ostream* os, const int8_t* ptr, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    if (i > 0) {
      *os << ", ";
    }
    *os << static_cast<int32_t>(ptr[i]); // NOLINT.
  }
}

template <>
void PrintValues<bool>(std::ostream* os, const bool* ptr, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    if (i > 0) {
      *os << ", ";
    }
    *os << (ptr[i] != 0 ? "true" : "false"); // NOLINT.
  }
}

void Constant::PrintData(std::ostream* os, size_t num_to_print) const {
  const Type& type = GetResultType();
  switch (type.GetDataType()) {
    case DataType::BOOL: {
      PrintValues(os, GetDataPtr<bool>(), num_to_print);
      break;
    }
    case DataType::INT8: {
      PrintValues(os, GetDataPtr<int8_t>(), num_to_print);
      break;
    }
    case DataType::UINT8: {
      PrintValues(os, GetDataPtr<uint8_t>(), num_to_print);
      break;
    }
    case DataType::INT32: {
      PrintValues(os, GetDataPtr<int>(), num_to_print);
      break;
    }
    case DataType::FLOAT32: {
      PrintValues(os, GetDataPtr<float>(), num_to_print);
      break;
    }
    case DataType::INT64: {
      PrintValues(os, GetDataPtr<int64_t>(), num_to_print);
      break;
    }
    default:
      HLCHECK(0 && "Unimplemented data type.");
  }
}

void Constant::Print(std::ostream& os) const {
  const Type& type = GetResultType();
  os << "Constant " << GetName() << "(";
  type.Print(os);
  os << ")";

  os << " = [";
  size_t num_of_elements = type.GetTotalNumOfElements();
  constexpr size_t limit = 32; // maximum number of elements to print.
  if (num_of_elements > 0) {
    PrintData(&os, std::min(num_of_elements, limit));
  }
  if (num_of_elements > limit) {
    os << ", ...";
  }
  os << "]\n";
}

bool Constant::IsScalarZero() const {
  const Type& type = GetResultType();
  return type.GetTotalNumOfElements() == 1 && HasSameValueOf(0);
}

bool Constant::IsScalarOne() const {
  const Type& type = GetResultType();
  return type.GetTotalNumOfElements() == 1 && HasSameValueOf(1);
}

bool Constant::HasSameValueOf(float x) const {
  const Type& type = GetResultType();
  for (int64_t i = 0, e = type.GetTotalNumOfElements(); i < e; ++i) {
    switch (type.GetDataType()) {
      case DataType::INT32: {
        if (GetData<int32_t>(i) != x) {
          return false;
        }
        break;
      }
      case DataType::FLOAT32: {
        if (GetData<float>(i) != x) {
          return false;
        }
        break;
      }
      case DataType::INT64: {
        if (GetData<int64_t>(i) != x) {
          return false;
        }
        break;
      }
      default: {
        return false;
      }
    }
  }
  return true;
}

template <typename T>
static T GetDataAs(const Constant& c, size_t idx) {
  const Type& type = c.GetResultType();
  switch (type.GetDataType()) {
    case DataType::INT32: {
      return (c.GetData<int32_t>(idx));
    }
    case DataType::FLOAT32: {
      return (c.GetData<float>(idx));
    }
    case DataType::INT64: {
      return (c.GetData<int64_t>(idx));
    }
    default: {
      return -1;
    }
  }
  return -1;
}

int64_t Constant::GetDataAsInt64(size_t idx) const {
  return GetDataAs<int64_t>(*this, idx);
}

float Constant::GetDataAsFloat32(size_t idx) const {
  return GetDataAs<float>(*this, idx);
}

} // namespace halo