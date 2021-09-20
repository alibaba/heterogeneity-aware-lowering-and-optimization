//===- transforms_util.h --------------------------------------------------===//
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

#include "halo/lib/ir/ir_builder.h"
#include "halo/lib/pass/pass.h"

#ifndef HALO_LIB_TRANSFORMS_TRANSFORMS_UTIL_H_
#define HALO_LIB_TRANSFORMS_TRANSFORMS_UTIL_H_

namespace halo {

template <typename T>
inline const T& GetAttributeValue(const Attribute& attr) {
  return T();
}

template <>
inline const std::string& GetAttributeValue(const Attribute& attr) {
  return attr.GetValueAsString();
}

template <>
inline const std::vector<int>& GetAttributeValue(const Attribute& attr) {
  return attr.GetValueAsIntegerList();
}

template <>
inline const std::vector<int64_t>& GetAttributeValue(const Attribute& attr) {
  return attr.GetValueAsInteger64List();
}

template <>
inline const std::vector<float>& GetAttributeValue(const Attribute& attr) {
  return attr.GetValueAsFloatList();
}

template <>
inline const std::vector<std::string>& GetAttributeValue(
    const Attribute& attr) {
  return attr.GetValueAsStringList();
}

template <>
inline const float& GetAttributeValue(const Attribute& attr) {
  return attr.GetValueAsFloat();
}

template <>
inline const bool& GetAttributeValue(const Attribute& attr) {
  return attr.GetValueAsBool();
}

template <>
inline const int& GetAttributeValue(const Attribute& attr) {
  return attr.GetValueAsInteger();
}

template <>
inline const DataType& GetAttributeValue(const Attribute& attr) {
  return attr.GetValueAsEnumDataType();
}

template <>
inline const TFIDFMode& GetAttributeValue(const Attribute& attr) {
  return attr.GetValueAsEnumTFIDF();
}

template <typename T>
const T& FindAttributeValue(const Instruction& inst, const std::string& name,
                            const T& default_val) {
  for (const auto& it : inst.GetAttributes()) {
    if (it->GetName() == name) {
      return GetAttributeValue<T>(*it);
    }
  }
  return default_val;
}

bool HasAttribute(const Instruction& inst, const std::string& name);

bool AppendReturnInst(BasicBlock* bb);

std::vector<int64_t> GetExtends(const std::vector<int64_t>& dims);

void SplitStringToInt64List(const std::string& src, std::vector<int64_t>* dst,
                            const std::string& delimiter);
template <typename T>
std::vector<Def> ConvertSqueezeImpl(const T* ext, IRBuilder* builder,
                                    const std::string& attr_name) {
  auto input = ext->GetOperand(0);
  const Type& input_type = input.GetType();

  if (!input_type.IsValid()) {
    return {};
  }

  std::vector<int32_t> squeeze_dims;
  HLCHECK(ext->GetNumOfAttributes() <= 1);
  if (ext->GetNumOfAttributes() == 1) {
    const Attribute* attr = ext->GetAttributes()[0].get();
    HLCHECK(attr->GetName() == attr_name);
    squeeze_dims = attr->GetValueAsIntegerList();
  }
  std::vector<int32_t> new_dims;
  for (size_t i = 0, e = input_type.GetNumOfDims(); i < e; ++i) {
    auto size = input_type.GetNumOfElementsInDim(i);
    if (size != 1) {
      new_dims.push_back(size);
    } else {
      if (!squeeze_dims.empty() &&
          std::find(squeeze_dims.begin(), squeeze_dims.end(), i) ==
              squeeze_dims.end()) {
        new_dims.push_back(size);
      }
    }
  }
  ConstantBuilder cb(ext->GetParent()->GetParent());
  Constant* c = cb.CreateConstant(
      ext->GetName() + "_squeeze_dims",
      Type{DataType::INT32, {static_cast<int64_t>(new_dims.size())}},
      new_dims.data());
  builder->SetInsertAfter(ext);
  auto new_inst = builder->CreateReshape(ext->GetName(), {input, *c});
  return {*new_inst};
}

} // end namespace halo.
#endif // HALO_LIB_TRANSFORMS_TRANSFORMS_UTIL_H_
