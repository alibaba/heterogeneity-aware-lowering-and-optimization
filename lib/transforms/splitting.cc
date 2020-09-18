//===- splitting.cc -------------------------------------------------------===//
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

#include "halo/lib/transforms/splitting.h"

#include <list>
#include <numeric>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "halo/lib/ir/ir_builder.h"

namespace halo {

struct InstNode;
using InstList = std::list<std::unique_ptr<InstNode>>;
struct InstNode {
  explicit InstNode(Instruction* inst) : Inst(inst) {}
  InstList Children;
  Instruction* Inst;
};

static std::unique_ptr<InstNode> GetSimpleSchedule(Function* func) {
  auto root = std::make_unique<InstNode>(nullptr);
  if (!func->IsEntryFunction()) {
    return root;
  }
  for (auto& bb : *func) {
    for (auto& ir : *bb) {
      if (!IsA<Instruction>(ir.get())) {
        continue;
      }
      Instruction* inst = DynCast<Instruction>(ir.get());
      if (inst->GetOpCode() == OpCode::CALL) {
        return root;
      }
      root->Children.push_back(std::make_unique<InstNode>(inst));
    }
  }

  InstNode* last_func = nullptr;
  for (auto it = root->Children.begin(), e = root->Children.end(); it != e;) {
    if ((*it)->Inst->GetOpCode() == OpCode::RETURN) {
      ++it;
      continue;
    }

    bool add_to_new_node = false;
    if (last_func == nullptr) {
      add_to_new_node = true;
    } else {
      Instruction* last_inst = last_func->Children.back()->Inst;
      if (last_inst != nullptr && last_inst->GetNumberOfUses() > 1) {
        add_to_new_node = true;
      }
    }
    if (!add_to_new_node) {
      int inst_op_num = 0;
      for (auto& op : (*it)->Inst->GetOperands()) {
        inst_op_num += IsA<Instruction>(op) ? 1 : 0;
      }
      if (inst_op_num > 1) {
        add_to_new_node = true;
      }
    }

    if (add_to_new_node) {
      auto node = std::make_unique<InstNode>((*it)->Inst);
      (*it)->Inst = nullptr;
      last_func = it->get();
      last_func->Children.push_back(std::move(node));
      ++it;
    } else {
      last_func->Children.push_back(std::move(*it));
      it = root->Children.erase(it);
    }
  }
  return root;
}

static Instruction* MoveInsts(Function* target_func,
                              const std::list<Instruction*>& insts) {
  BasicBlock* bb;
  if (target_func->BasicBlocks().empty()) {
    BasicBlockBuilder bb_builder(target_func);
    bb = bb_builder.CreateBasicBlock("bb0");
  } else {
    bb = target_func->BasicBlocks().back().get();
  }

  std::unordered_set<IRObject*> irs(insts.begin(), insts.end());
  std::unordered_set<Constant*> constants;
  std::unordered_set<Def> input_set;
  std::vector<Def> inputs; // make sure the iterator order is determinstic.
  std::vector<Instruction*> outputs{insts.back()};
  std::unordered_map<Def, Def> op_mapping;

  // determine inputs/outputs
  for (auto inst : insts) {
    for (auto& op : inst->GetOperands()) {
      IRObject* def = op.GetDef();
      if (irs.count(def) != 0 || input_set.count(op) != 0) {
        continue;
      }
      if (IsA<Constant>(def)) {
        constants.insert(DynCast<Constant>(def));
      } else {
        input_set.insert(op);
        inputs.push_back(op);
      }
    }
  }

  // Insert call at call site.
  Instruction* insert_pos = outputs.back();
  auto orig_ir_builder = IRBuilder(insert_pos->GetParent());
  orig_ir_builder.SetInsertAfter(insert_pos);
  auto call = orig_ir_builder.CreateCall("call_" + target_func->GetName(),
                                         {inputs.begin(), inputs.end()});
  int num_of_results = std::accumulate(
      outputs.begin(), outputs.end(), 0,
      [](int a, Instruction* inst) { return a + inst->GetNumOfResults(); });
  call->SetCallee(target_func);
  call->SetNumOfResults(num_of_results);

  // Make arguments
  if (!inputs.empty()) {
    ArgumentBuilder arg_builder(target_func);
    for (const auto& in : inputs) {
      auto arg = arg_builder.CreateArgument(
          in.GetOwner()->GetName() + "_" + std::to_string(in.GetIdx()),
          in.GetType());
      op_mapping.insert({in, Def(arg, 0)});
    }
  }

  // Make constants
  if (!constants.empty()) {
    ConstantBuilder cb(target_func);
    for (const auto& c : constants) {
      auto new_c = cb.CreateConstant(c->GetName(), c->GetResultType(),
                                     c->GetRawDataPtr());
      op_mapping.insert({Def(c, 0), Def(new_c, 0)});
    }
  }

  // Clone instrs.
  IRBuilder ir_builder(bb);
  for (auto inst : insts) {
    std::vector<Def> ops;
    ops.reserve(inst->GetNumOfOperands());
    for (const auto& op : inst->GetOperands()) {
      auto it = op_mapping.find(op);
      HLCHECK(it != op_mapping.end());
      ops.push_back(it->second);
    }
    auto new_inst = ir_builder.Clone(*inst, ops);
    for (int i = 0, e = inst->GetNumOfResults(); i < e; ++i) {
      op_mapping.insert({Def(inst, i), Def(new_inst, i)});
    }
  }
  // Insert Return
  auto ret_inst = ir_builder.CreateReturn("ret", std::vector<Def>{});

  int idx = 0;
  for (auto out_inst : outputs) {
    int n = out_inst->GetNumOfResults();
    for (int i = 0; i < n; ++i) {
      auto it = op_mapping.find(Def{out_inst, i});
      HLCHECK(it != op_mapping.end());
      ret_inst->AddOneOperand(it->second);
      out_inst->ReplaceAllUsesWith(i, Def{call, idx});
      call->GetResultsTypes()[idx] = out_inst->GetResultsTypes()[i];
      ++idx;
    }
  }
  return call;
}

static Instruction* ScheduleToFunctions(Function* caller, InstNode* parent,
                                        InstNode* curr) {
  std::list<Instruction*> insts;
  int idx = 0;
  FunctionBuilder fb(caller->GetParent());
  // visit leaves first.
  for (auto& c : curr->Children) {
    auto inst = c->Inst;
    if (!c->Children.empty()) {
      HLCHECK(inst == nullptr);
      auto new_func =
          fb.CreateFunction(caller->GetName() + "_" + std::to_string(idx++));
      inst = ScheduleToFunctions(new_func, curr, c.get());
      HLCHECK(inst->GetOpCode() == OpCode::CALL);
    }
    insts.push_back(inst);
  }
  if (parent != nullptr) {
    return MoveInsts(caller, insts);
  }
  return nullptr;
}

bool Splitting::RunOnFunction(Function* func) {
  auto sched = GetSimpleSchedule(func);

  return ScheduleToFunctions(func, nullptr, sched.get()) != nullptr;
}

} // end namespace halo