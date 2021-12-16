//===- convert_tf_cfg.cc --------------------------------------------------===//
//
// Copyright (C) 2019-2021 Alibaba Group Holding Limited.
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

#include "halo/lib/transforms/convert_tf_cfg.h"

#include <iterator>
#include <set>

#include "halo/api/halo_data.h"
#include "halo/lib/ir/controlflow_instructions.h"
#include "halo/lib/ir/extension_instructions.h"
#include "halo/lib/ir/instruction.h"
#include "halo/lib/ir/ir_builder.h"

namespace halo {

// Merge multiple "if" instructions with the same condition into one.
// Initially, each true/false branch has at most one instruction when
// they are converted from TF Switch directly.
static bool MergeIfs(BasicBlock* bb) {
  bool changed = false;
  std::unordered_map<Def, IfInst*> ifs;
  for (const auto& it : *bb) {
    IfInst* inst = DynCast<IfInst>(it.get());
    if (inst == nullptr) {
      continue;
    }
    const Def& cond = inst->GetOperand(0);
    if (ifs.count(cond) == 0) {
      ifs[cond] = inst;
      continue;
    }

    IfInst* dst = ifs[cond];
    for (int i = 1, e = inst->GetNumOfOperands(); i < e; ++i) {
      dst->AddOneOperand(inst->GetOperand(i));
    }
    inst->DropAllOperands();
    for (unsigned i = 0, dst_idx = dst->GetNumOfResults();
         i < inst->GetNumOfResults(); ++i) {
      const auto& ty = inst->GetResultsTypes()[i];
      dst->GetResultsTypes().push_back(ty);
      inst->ReplaceAllUsesWith(i, Def{dst, static_cast<int>(dst_idx++)});
    }
    inst->GetThenBranch()->MoveTo(dst->GetThenBranch());
    inst->GetElseBranch()->MoveTo(dst->GetElseBranch());
    // merge arguments
    auto merge_args = [](BasicBlock* bb) {
      if (bb->GetNumOfOperands() == 0) { // who removed args?
        return;
      }
      HLCHECK(bb->GetNumOfOperands() == 2);
      Def arg0{bb->arg_front(), 0};
      bb->arg_back()->ReplaceAllUsesWith({arg0});
      bb->Args().pop_back();
    };
    merge_args(inst->GetThenBranch());
    merge_args(inst->GetElseBranch());

    changed = true;
  }
  return changed;
}

static void RewriteOutput(IfInst* if_inst, const std::vector<Def>& ops,
                          bool is_taken) {
  auto bb = is_taken ? if_inst->GetThenBranch() : if_inst->GetElseBranch();
  ReturnInst* ret = bb->GetReturnInst();
  HLCHECK(ret != nullptr);
  ret->DropAllOperands();
  HLCHECK(if_inst->GetNumOfResults() == (if_inst->GetNumOfOperands() - 1) * 2);
  for (auto op : ops) {
    // if's output: [v1_f, v1_t, v2_f, v2_t, ...], inputs: [cond, v1, v2, v3]
    // Branch bb's args: [arg1, arg2, arg3]
    if (op.GetOwner() == if_inst) {
      op = Def{std::next(bb->arg_begin(), op.GetIdx() / 2)->get(), 0};
    }
    ret->AddOneOperand(op);
  }
}

static bool RunOnBasicBlock(BasicBlock* bb) {
  // run on main bb only. Fixme: need to deal with nested if.
  if (bb != bb->GetParent()->begin()->get()) {
    return false;
  }
  bool changed = false;
  changed |= MergeIfs(bb);
  std::unordered_map<BasicBlock*, IfInst*> branch_bbs;
  for (const auto& it : *bb) {
    IfInst* inst = DynCast<IfInst>(it.get());
    if (inst != nullptr) {
      branch_bbs[inst->GetThenBranch()] = inst;
      branch_bbs[inst->GetElseBranch()] = inst;
    }
  }

  for (const auto& it : *bb) {
    Instruction* inst = it.get();
    // tf_merge will be handled later.
    if (auto ext = DynCast<TFExtensionInst>(inst);
        ext != nullptr && ext->GetExtOpCode() == TFExtOpCode::MERGE) {
      continue;
    }

    // If an instruction uses a value that is only defined in some branch,
    // move it to that branch's BB. It's illegal for one instruction uses valued
    // defined in both branch.
    BasicBlock* new_parent = nullptr;
    for (int i = 0, e = inst->GetNumOfOperands(); i < e; ++i) {
      const auto& op = inst->GetOperand(i);
      auto if_inst = DynCast<IfInst>(op);
      if (if_inst != nullptr) {
        int idx = op.GetIdx();
        auto bb = (idx & 1) == 0 ? if_inst->GetElseBranch()
                                 : if_inst->GetThenBranch();
        if (new_parent == nullptr) {
          new_parent = bb;
        } else {
          HLCHECK(new_parent == bb);
        }
      } else {
        Instruction* op_inst = DynCast<Instruction>(op);
        BasicBlock* op_bb = op_inst == nullptr ? nullptr : op_inst->GetParent();
        if (branch_bbs.count(op_bb) > 0) {
          if (new_parent == nullptr) {
            new_parent = op_bb;
          } else {
            HLCHECK(new_parent == op_bb);
          }
        }
      }
    }
    if (new_parent != nullptr) {
      IfInst* if_inst = branch_bbs[new_parent];
      HLCHECK(if_inst != nullptr);
      IRBuilder new_builder(new_parent);
      new_builder.SetInsertBefore(new_parent->GetReturnInst());
      std::vector<Def> operands = inst->GetOperands();
      for (auto& op : operands) {
        if (op.GetOwner() == if_inst) {
          op = Def{std::next(new_parent->arg_begin(), op.GetIdx() / 2)->get(),
                   0};
        }
      }
      auto new_inst = new_builder.Clone(*inst, operands);
      new_inst->GetResultsTypes() = inst->GetResultsTypes();
      HLCHECK(new_inst->GetOpCode() != OpCode::RETURN);
      for (int i = 0, e = inst->GetNumOfResults(); i < e; ++i) {
        inst->ReplaceAllUsesWith(i, Def{new_inst, i});
      }
      changed |= true;
    }
  }

  // Merge multiple tf_merge that associates with same if.
  // All the inputs of tf_merge should associate with the same if.
  std::unordered_map<IfInst*, std::vector<TFExtensionInst*>> if2merge;

  for (const auto& it : *bb) {
    TFExtensionInst* inst = DynCast<TFExtensionInst>(it.get());
    if (inst == nullptr || inst->GetExtOpCode() != TFExtOpCode::MERGE) {
      continue;
    }
    IfInst* if_inst = nullptr;
    for (auto& op : inst->GetOperands()) {
      Instruction* op_inst = DynCast<Instruction>(op);
      if (op_inst->GetOpCode() == OpCode::IF) {
        // some branch is empty. nested if?
        HLCHECK(if_inst == nullptr || if_inst == op_inst);
        if_inst = DynCast<IfInst>(op_inst);
      } else {
        BasicBlock* bb = op_inst->GetParent();
        auto it = branch_bbs.find(bb);
        HLCHECK(it != branch_bbs.end());
        HLCHECK(if_inst == nullptr || if_inst == it->second);
        if_inst = it->second;
      }
    }
    HLCHECK(if_inst != nullptr);
    if2merge[if_inst].push_back(inst);
  }
  for (auto& if_merge : if2merge) {
    std::vector<Def> true_ops;
    std::vector<Def> false_ops;
    IfInst* if_inst = if_merge.first;
    std::set<int> op_indices;
    for (Instruction* merge : if_merge.second) {
      for (auto& op : merge->GetOperands()) {
        bool is_taken = false;
        if (op.GetOwner() == if_inst) {
          is_taken = (op.GetIdx() & 1) == 1;
        } else {
          const Instruction* inst = DynCast<Instruction>(op.GetOwner());
          HLCHECK(inst != nullptr);
          const auto& bb = inst->GetParent();
          auto it = branch_bbs.find(bb);
          HLCHECK(branch_bbs.end() != it);
          HLCHECK(bb == it->second->GetThenBranch() ||
                  bb == it->second->GetElseBranch());
          is_taken = bb == it->second->GetThenBranch();
        }
        if (is_taken) {
          true_ops.push_back(op);
        } else {
          false_ops.push_back(op);
        }
      }
    }
    HLCHECK(true_ops.size() == false_ops.size());

    RewriteOutput(if_inst, true_ops, true);
    RewriteOutput(if_inst, false_ops, false);
    std::vector<Type> rets;
    rets.reserve(true_ops.size());
    for (int i = 0, e = true_ops.size(); i < e; ++i) {
      const auto& true_ty = true_ops[i].GetType();
      const auto& false_ty = false_ops[i].GetType();
      // The output type is dynamic. Here we just pick a valid one.
      rets.push_back(true_ty.IsValid() ? true_ty : false_ty);
    }
    if_inst->GetResultsTypes() = rets;
    for (int i = 0, e = if_merge.second.size(); i < e; ++i) {
      if_merge.second[i]->ReplaceAllUsesWith({Def{if_inst, i}});
    }
  }
  // Modify TF_Merge and associated "if":
  // Before:
  //   if_results(true_val, false_val) = if (...)
  //   out = merge(if_results)
  // After:
  //   if_result(val) = if(...)
  //   out = val

  return changed;
}

bool ConvertTFCFG::RunOnFunction(Function* func) {
  bool changed = false;
  if (converted_) {
    return false;
  }
  for (auto it = func->begin(), e = func->end(); it != e;) {
    BasicBlock* bb = it->get();
    if (bb->Instructions().empty()) {
      it = func->BasicBlocks().erase(it);
      continue;
    }
    changed |= RunOnBasicBlock(bb);
    it = std::next(it);
  }
  converted_ = true;
  return changed;
}

} // end namespace halo
