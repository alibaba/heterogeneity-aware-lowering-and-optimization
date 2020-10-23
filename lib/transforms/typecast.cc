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

bool TypeCast::RunOnFunction(Function* func) {
  bool changed = false;
  // Replace constants.
  ConstantBuilder cb(func);
  Function::ConstantList& constants = func->Constants();
  for (auto it = constants.begin(), ie = constants.end(); it != ie; ++it) {
    const auto& orig_type = (*it)->GetResultType(0);
    if (orig_type.GetDataType() == DataType::INT64) {
      std::vector<int> ret;
      ret.reserve(orig_type.GetTotalNumOfElements());
      for (unsigned int i = 0; i < orig_type.GetTotalNumOfElements(); ++i) {
        ret.push_back(static_cast<int>((*it)->GetData<int64_t>(i)));
      }
      Constant* c_ret = cb.CreateConstant(
          (*it)->GetName() + "_castdown",
          halo::Type{DataType::INT32, orig_type.GetDimSizes()}, ret.data());
      (*it)->ReplaceAllUsesWith(0, *c_ret);
      changed = true;
    }
  }

  for (auto& bb : *func) {
    for (auto& inst : *bb) {
      for (unsigned int i = 0; i < inst->GetNumOfResults(); ++i) {
        const auto& orig_type = inst->GetResultType(i);
        if (orig_type.IsValid() && orig_type.GetDataType() == DataType::INT64) {
          inst->GetResultsTypes()[i] =
              halo::Type{DataType::INT32, orig_type.GetDimSizes()};
        }
      }
    }
  }
  return changed;
} // namespace halo

} // end namespace halo
