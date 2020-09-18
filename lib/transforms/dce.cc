//===- dce.cc -------------------------------------------------------------===//
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

#include "halo/lib/transforms/dce.h"

#include <iterator>

#include "halo/lib/ir/ir_builder.h"

namespace halo {

bool DCE::RunOnBasicBlock(BasicBlock* bb) {
  bool changed = false;
  for (BasicBlock::reverse_iterator it = bb->rbegin(), e = bb->rend();
       it != e;) {
    Instruction* inst = it->get();
    if (inst->GetOpCode() == OpCode::RETURN || inst->GetNumberOfUses() != 0) {
      ++it;
      continue;
    }
    changed = true;
    auto nit = std::next(it);
    // Delete the instruction.
    inst->DropAllOperands();
    bb->Instructions().erase(nit.base());
    it = nit;
  }

  auto remove = [](auto& objs) {
    bool changed = false;
    for (auto it = objs.begin(), ie = objs.end(); it != ie;) {
      if ((*it)->GetNumberOfUses() == 0) {
        it = objs.erase(it);
        changed = true;
      } else {
        it = std::next(it);
      }
    }
    return changed;
  };

  // Remove dead constants.
  auto func = bb->GetParent();
  Function::ConstantList& constants = func->Constants();
  changed |= remove(constants);

  // Remove dead arguments.
  auto& args = func->Args();
  changed |= remove(args);
  return changed;
}

} // end namespace halo
