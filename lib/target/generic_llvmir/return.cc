//===- return.cc ----------------------------------------------------------===//
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

#include <cstdio>

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Support/Casting.h"
#include "halo/lib/ir/ir_builder.h"
#include "halo/lib/target/generic_llvmir/generic_llvmir_codegen.h"

namespace halo {

void GenericLLVMIRCodeGen::RunOnInstruction(ReturnInst* inst) {
  llvm::IRBuilder<>* ir_builder = current_llvm_builder_;

  size_t idx = 0;
  for (auto& op : inst->GetOperands()) {
    llvm::Value* val = ir_mapping_[op];
    llvm::Value* ptr = ir_mapping_[Def(inst, idx++)];
    if (val->getType()->isPointerTy()) {
      auto elem_type = val->getType()->getPointerElementType();
      if (elem_type->isVectorTy() || elem_type->isArrayTy()) {
        elem_type =
            llvm::cast<llvm::SequentialType>(elem_type)->getElementType();
      }
      auto alignment =
          llvm_module_->getDataLayout().getABITypeAlignment(elem_type);
      auto elem_size =
          llvm_module_->getDataLayout().getTypeAllocSize(elem_type);
      ir_builder->CreateMemCpy(
          ptr, alignment, val, alignment,
          ir_builder->getInt64(op.GetType().GetTotalNumOfElements() *
                               elem_size));
    } else {
      ir_builder->CreateStore(val, ptr);
    }
  }
}

} // namespace halo