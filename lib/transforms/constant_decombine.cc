//===- constant_decombine.cc ----------------------------------------------===//
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

#include "halo/lib/transforms/constant_decombine.h"

#include <array>
#include <iterator>

#include "halo/lib/ir/ir_builder.h"

namespace halo {

// find all constant nodes if their users are great than 1, then duplicate
// constant node and assign a new unique name, replace operands with new
// constant
bool ConstantDecombine::RunOnFunction(Function* func) {
  bool changed = false;
  // Duplicate constants.
  ConstantBuilder cb(func);
  Function::ConstantList& constants = func->Constants();
  for (auto it = constants.begin(), ie = constants.end(); it != ie; ++it) {
    std::list<Use> uses = (*it)->GetIthResultUses(0).GetUses();
    size_t num_of_uses = uses.size();
    if (num_of_uses > 1) {
      int i = 0;
      std::vector<Constant*> constant_copy(num_of_uses, nullptr);
      VLOG(1) << "constant node name: " << (*it)->GetName();
      for (auto& u : uses) {
        VLOG(1) << "use operand idx: " << u.GetUseOperandIdx();
        // u.ReplaceAllUsesWith(u.GetUseOperandIdx(), Def{constant_copy[i], 0});
        constant_copy[i] = cb.Clone(*(*it));
        constant_copy[i]->SetName((*it)->GetName() + "_copy_" +
                                  std::to_string(i));
        constant_copy[i]->GetResultsTypes()[0] = (*it)->GetResultType();
        auto used_by = u.GetOwner();
        VLOG(1) << "used_by node name: " << used_by->GetName();
        used_by->ReplaceOperandWith(u.GetUseOperandIdx(),
                                    Def{constant_copy[i], 0});
        i++;
      }
      changed = true;
    }
  }
  return changed;
}

} // end namespace halo
