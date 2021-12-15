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
#include <set>

#include "halo/lib/ir/ir_builder.h"

namespace halo {

static void RemoveBody(BasicBlock* bb) {
  auto return_inst = bb->GetReturnInst();
  if (return_inst != nullptr) {
    // Drop all the operands of the return instruction so the rest of the body
    // loop will be DCE'ed automatically.
    // Note that the return inst cannot be erased because the current legalizer
    // will try to append one if no return inst exists for a block.
    return_inst->DropAllOperands();
    if (bb->Instructions().size() == 1) {
      bb->Instructions().clear();
      return;
    }
  }
}
static void RemoveLoopBody(LoopInst* loop_inst) {
  RemoveBody(loop_inst->GetBody());
}

static void RemoveIfBody(IfInst* if_inst) {
  RemoveBody(if_inst->GetThenBranch());
  RemoveBody(if_inst->GetElseBranch());
}

// For instructions with `undef` operands, they are unreachable except for
// `tf_merge` and optional operands.
static bool RemoveUndefInstrs(BasicBlock* bb) {
  bool changed = false;
  const auto& undef = Def::GetUndefined();
  for (auto it = bb->begin(), e = bb->end(); it != e; ++it) {
    Instruction* inst = it->get();
    if (auto ext = DynCast<TFExtensionInst>(inst);
        inst->GetOpCode() == OpCode::RETURN ||
        (ext != nullptr && ext->GetExtOpCode() == TFExtOpCode::MERGE)) {
      continue;
    }
    bool has_undef = false;
    for (int i = 0, e = inst->GetNumOfOperands(); !has_undef && i < e; ++i) {
      has_undef = !inst->IsOperandOptional(i) && inst->GetOperand(i).IsNull();
    }
    if (!has_undef) {
      continue;
    }
    for (int idx = 0, e = inst->GetNumOfResults(); idx < e; ++idx) {
      inst->ReplaceAllUsesWith(idx, undef);
    }
    changed |= true;
  }
  return changed;
}

bool DCE::RunOnBasicBlock(BasicBlock* bb) {
  bool changed = false;
  changed |= RemoveUndefInstrs(bb);
  std::set<Instruction*> dead_instrs;
  for (BasicBlock::reverse_iterator it = bb->rbegin(), e = bb->rend(); it != e;
       ++it) {
    Instruction* inst = it->get();
    if (inst->GetOpCode() == OpCode::RETURN || inst->GetNumberOfUses() != 0) {
      continue;
    }
    changed = true;
    inst->DropAllOperands();
    dead_instrs.insert(inst);
  }

  // Delete the dead instructions.
  for (auto it = bb->begin(), e = bb->end(); it != e;) {
    Instruction* inst = it->get();
    if (dead_instrs.count(inst) > 0) {
      if (inst->GetOpCode() == OpCode::LOOP) {
        RemoveLoopBody(DynCast<LoopInst>(inst));
      }
      if (inst->GetOpCode() == OpCode::IF) {
        RemoveIfBody(DynCast<IfInst>(inst));
      }
      it = bb->Instructions().erase(it);
    } else {
      it = std::next(it);
    }
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
