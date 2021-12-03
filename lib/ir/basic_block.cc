//===- basic_block.cc -----------------------------------------------------===//
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

#include "halo/lib/ir/basic_block.h"

#include "halo/lib/ir/common_instructions.h"
#include "halo/lib/ir/ir_builder.h"

namespace halo {

ReturnInst* BasicBlock::GetReturnInst() const {
  for (auto it = instructions_.rbegin(), ie = instructions_.rend(); it != ie;
       ++it) {
    if ((*it)->GetOpCode() == OpCode::RETURN) {
      return DynCast<ReturnInst>(it->get());
    }
  }
  return nullptr;
}

void BasicBlock::MoveTo(BasicBlock* dst) noexcept {
  // Move args into dst.
  dst->args_.splice(dst->args_.end(), args_);

  // Combine Return operands and remove ReturnInst from both blocks.
  Instruction* dst_ret = dst->GetReturnInst();
  Instruction* curr_ret = GetReturnInst();
  std::vector<Def> ret_ops;
  std::vector<Type> ret_types;
  std::string ret_name = "ret";
  if (dst_ret != nullptr) {
    ret_name = dst_ret->GetName();
    ret_ops = dst_ret->GetOperands();
    ret_types = dst_ret->GetResultsTypes();
    dst_ret->DropAllOperands();
    dst->instructions_.pop_back();
  }
  if (curr_ret != nullptr) {
    ret_ops.insert(ret_ops.end(), curr_ret->GetOperands().begin(),
                   curr_ret->GetOperands().end());
    ret_types.insert(ret_types.end(), curr_ret->GetResultsTypes().begin(),
                     curr_ret->GetResultsTypes().end());
    curr_ret->DropAllOperands();
    instructions_.pop_back();
  }

  // Move current instructions into dst.
  dst->instructions_.splice(dst->instructions_.end(), instructions_);
  IRBuilder builder(dst);
  if (!dst->empty()) {
    builder.SetInsertAfter(dst->back());
  }

  // Create new return at end.
  builder.CreateReturn("ret", ret_ops);
}

void BasicBlock::Print(std::ostream& os) const {
  os << "BasicBlock: " << GetName() << "(";
  int arg_idx = 0;
  for (auto& arg : Args()) {
    if (arg_idx++ != 0) {
      os << ", ";
    }
    arg->Print(os);
  }
  os << ")\n";

  for (auto& c : Constants()) {
    c->Print(os);
  }
  for (const auto& inst : *this) {
    os << "  ";
    inst->Print(os);
  }
}

} // namespace halo
