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
#include <climits>
#include <cmath>
#include <iostream>
#include <variant>

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

Constant::Constant(GlobalContext& context, const std::string& name,
                   const Type& type, const std::vector<std::string>& strings)
    : IRObject(context, name, 1),
      parent_(nullptr),
      data_layout_(context.GetDefaultDataLayout()) {
  HLCHECK(type.GetDataType() == DataType::STRING);
  HLCHECK(type.IsValid());
  SetData(type, strings);
}

Constant::Constant(const Constant& from)
    : IRObject(from.GetGlobalContext(), from.GetName(), 1),
      parent_(nullptr),
      data_layout_(from.data_layout_),
      data_(from.data_),
      string_data_(from.string_data_) {}

struct Float {
  // TODO(unknown): no infinity, underflow/overflow handling.
 private:
  Float() = delete;
  static constexpr int BitsPerInt = CHAR_BIT * sizeof(int);
  template <typename T, int exp, int mantissa>
  static std::array<int, 3> Extract(T x) {
    static_assert(exp + mantissa + 1 == sizeof(T) * CHAR_BIT);
    int sign = x >> (exp + mantissa);
    int m = x & ((1 << mantissa) - 1);
    int e = (x >> mantissa) & ((1 << exp) - 1);
    return {sign, e, m};
  }

  template <typename T, int exp, int mantissa>
  static T Combine(int sign, int e, int m) {
    static_assert(exp + mantissa + 1 == sizeof(T) * CHAR_BIT);
    T x{0};
    x = sign ? 1U << (exp + mantissa) : 0;
    m >>= BitsPerInt - mantissa;
    x |= m & ((1U << mantissa) - 1);
    x |= (e & ((1U << exp) - 1)) << mantissa;
    return x;
  }
  static constexpr int FP32Exp = 8;
  static constexpr int FP32Mantissa = 23;
  static constexpr int FP32ExpBias = 127;
  static constexpr int FP16Exp = 5;
  static constexpr int FP16Mantissa = 10;
  static constexpr int FP16ExpBias = 15;

  static inline float GetFP32(uint8_t sign, int32_t e, uint32_t m) {
    uint32_t x =
        Combine<uint32_t, FP32Exp, FP32Mantissa>(sign, e + FP32ExpBias, m);
    const void* p = &x;
    return *(reinterpret_cast<const float*>(p)); // NOLINT.
  }
  static inline uint16_t GetFP16(uint8_t sign, int32_t e, uint32_t m) {
    return Combine<uint16_t, FP16Exp, FP16Mantissa>(sign, e + FP16ExpBias, m);
  }

 public:
  static inline uint16_t GetFP16(float x) {
    const void* p = &x;
    uint32_t v = *(reinterpret_cast<const int*>(p)); // NOLINT.
    auto components = Extract<uint32_t, FP32Exp, FP32Mantissa>(v);
    components[1] -= FP32ExpBias;
    components[2] <<= BitsPerInt - FP32Mantissa;
    return GetFP16(components[0], components[1], components[2]);
  }

  static inline float GetFP32(uint16_t x) {
    auto components = Extract<uint16_t, FP16Exp, FP16Mantissa>(x);
    components[1] -= FP16ExpBias;
    components[2] <<= BitsPerInt - FP16Mantissa;
    return GetFP32(components[0], components[1], components[2]);
  }
};

template <typename T>
static void PrintValues(std::ostream* os, const T* ptr, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    if (i > 0) {
      *os << ", ";
    }
    *os << ptr[i]; // NOLINT.
  }
}

template <typename T>
static void PrintFPValue(std::ostream* os, const T& x) {
  if (std::isnan(x)) {
    *os << "NAN";
    return;
  }
  if (std::isinf(x)) {
    if (x < 0) {
      *os << "-";
    }
    *os << "INFINITY";
    return;
  }
  *os << x;
}

template <>
void PrintValues<float>(std::ostream* os, const float* ptr, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    if (i > 0) {
      *os << ", ";
    }
    PrintFPValue(os, ptr[i]); // NOLINT.
  }
}

template <>
void PrintValues<double>(std::ostream* os, const double* ptr, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    if (i > 0) {
      *os << ", ";
    }
    PrintFPValue(os, ptr[i]); // NOLINT.
  }
}

static void PrintFP16Values(std::ostream* os, const uint16_t* ptr, size_t n,
                            bool human_friendly) {
  for (size_t i = 0; i < n; ++i) {
    if (i > 0) {
      *os << ", ";
    }
    if (human_friendly) {
      *os << Float::GetFP32(ptr[i]); // NOLINT.
      ;
    } else {
      *os << ptr[i]; // NOLINT.
    }
  }
}

static void PrintBF16Values(std::ostream* os, const uint16_t* ptr, size_t n,
                            bool human_friendly) {
  for (size_t i = 0; i < n; ++i) {
    if (i > 0) {
      *os << ", ";
    }
    constexpr int shift_amt = 16;
    int ui32 = ptr[i] << shift_amt; // NOLINT.
    const void* p = &ui32;
    *os << *(reinterpret_cast<const float*>(p)); // NOLINT.
  }
}

void Constant::SetData(const Type& ty,
                       const std::vector<std::string>& strings) {
  HLCHECK(ty.GetDataType() == DataType::STRING);
  auto& results = GetResultsTypes();
  results.resize(1);
  results[0] = ty;
  string_data_ = strings;
}

void Constant::SetData(const Type& ty, const void* data_ptr, bool do_splat) {
  HLCHECK(ty.GetDataType() != DataType::STRING);
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

static void PrintValues(std::ostream* os,
                        const std::vector<std::string>& strings, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    if (i > 0) {
      *os << ", ";
    }
    *os << '"' << strings[i] << '"';
  }
}

void Constant::PrintData(std::ostream* os, size_t num_to_print,
                         bool human_friendly) const {
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
    case DataType::INT16: {
      PrintValues(os, GetDataPtr<int16_t>(), num_to_print);
      break;
    }
    case DataType::UINT16: {
      PrintValues(os, GetDataPtr<uint16_t>(), num_to_print);
      break;
    }
    case DataType::INT32: {
      PrintValues(os, GetDataPtr<int>(), num_to_print);
      break;
    }
    case DataType::FLOAT16: {
      PrintFP16Values(os, static_cast<const uint16_t*>(GetRawDataPtr()),
                      num_to_print, human_friendly);
      break;
    }
    case DataType::BFLOAT16: {
      PrintBF16Values(os, static_cast<const uint16_t*>(GetRawDataPtr()),
                      num_to_print, human_friendly);
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
    case DataType::FLOAT64: {
      PrintValues(os, GetDataPtr<double>(), num_to_print);
      break;
    }
    case DataType::STRING: {
      PrintValues(os, string_data_, num_to_print);
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
    PrintData(&os, std::min(num_of_elements, limit), true);
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

std::vector<int64_t> Constant::GetDataAsInt64() const {
  auto n = GetResultType().GetTotalNumOfElements();
  std::vector<int64_t> ret(n);
  for (int i = 0; i != n; ++i) {
    ret[i] = GetDataAsInt64(i);
  }
  return ret;
}

float Constant::GetDataAsFloat32(size_t idx) const {
  return GetDataAs<float>(*this, idx);
}

} // namespace halo