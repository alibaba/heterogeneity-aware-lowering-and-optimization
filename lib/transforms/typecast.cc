//===- castdown_int64.cc --------------------------------------------------===//
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

#include "halo/lib/transforms/typecast.h"

#include <iterator>

#include "halo/lib/ir/ir_builder.h"

namespace halo {

template <typename Tsrc, typename Tdst>
static bool ReplaceConstant(Constant* c, DataType src_type, DataType new_type) {
  const auto& orig_type = c->GetResultType(0);
  if (orig_type.GetDataType() != src_type) {
    return false;
  }
  std::vector<Tdst> ret;
  auto n = orig_type.GetTotalNumOfElements();
  ret.reserve(n);
  for (unsigned int i = 0; i < n; ++i) {
    ret.push_back(static_cast<float>(c->GetData<Tsrc>(i)));
  }
  c->SetData(halo::Type{new_type, orig_type.GetDimSizes()}, ret.data());
  return true;
}

bool TypeCast::RunOnFunction(Function* func) {
  bool changed = false;
  // Replace arguments.
  for (auto& arg : func->Args()) {
    const auto& ty = arg->GetResultType();
    if (ty.GetDataType() == DataType::INT64) {
      halo::Type new_ty{DataType::INT32, ty.GetDimSizes()};
      arg->SetType(new_ty);
      changed |= true;
    } else if (ty.GetDataType() == DataType::FLOAT64) {
      halo::Type new_ty{DataType::FLOAT32, ty.GetDimSizes()};
      arg->SetType(new_ty);
      changed |= true;
    }
  }

  // Replace constants.
  Module* m = func->GetParent();
  for (auto& c : m->Constants()) {
    changed |= ReplaceConstant<double, float>(c.get(), DataType::FLOAT64,
                                              DataType::FLOAT32);
    changed |= ReplaceConstant<int64_t, int32_t>(c.get(), DataType::INT64,
                                                 DataType::INT32);
  }

  for (auto& c : func->Constants()) {
    changed |= ReplaceConstant<double, float>(c.get(), DataType::FLOAT64,
                                              DataType::FLOAT32);
    changed |= ReplaceConstant<int64_t, int32_t>(c.get(), DataType::INT64,
                                                 DataType::INT32);
  }

  for (auto& bb : *func) {
    for (auto& inst : *bb) {
      for (unsigned int i = 0; i < inst->GetNumOfResults(); ++i) {
        const auto& orig_type = inst->GetResultType(i);
        if (orig_type.IsValid() && orig_type.GetDataType() == DataType::INT64) {
          inst->GetResultsTypes()[i] =
              halo::Type{DataType::INT32, orig_type.GetDimSizes()};
        } else if (orig_type.IsValid() &&
                   orig_type.GetDataType() == DataType::FLOAT64) {
          inst->GetResultsTypes()[i] =
              halo::Type{DataType::FLOAT32, orig_type.GetDimSizes()};
        }
      }
    }
  }
  return changed;
}

} // end namespace halo
