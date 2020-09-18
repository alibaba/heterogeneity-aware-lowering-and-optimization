//===- math_unary.cc ------------------------------------------------------===//
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

#include "halo/lib/ir/ir_builder.h"
#include "halo/lib/target/generic_llvmir/generic_llvmir_codegen.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/IRBuilder.h"

namespace halo {

void GenericLLVMIRCodeGen::RunOnMathUnaryInstruction(Instruction* inst) {
  llvm::IRBuilder<>* ir_builder = current_llvm_builder_;
  const Def& lhs = inst->GetOperand(0);

  llvm::Value* op0 = ir_mapping_[lhs];

  auto lhs_elms = lhs.GetType().GetTotalNumOfElements();

  // TODO(unknown): we can split into multiple smaller vec ops.
  llvm::PointerType* data_ptr_type =
      SNTypeToLLVMType(lhs.GetType().GetDataType())->getPointerTo();
  llvm::Type* i64_type = ir_builder->getInt64Ty();

  llvm::FunctionType* ftype = llvm::FunctionType::get(
      ir_builder->getVoidTy(), {data_ptr_type, data_ptr_type, i64_type}, false);

  std::string fname = GetRTLibFuncName(*inst, lhs.GetType().GetDataType());

  llvm::FunctionCallee callee = llvm_module_->getOrInsertFunction(fname, ftype);

  llvm::Value* ret_buf = AllocateLLVMBuffer(ir_builder, Def{inst, 0});
  auto ret_buf_ptr = ir_builder->CreateBitCast(ret_buf, data_ptr_type);
  op0 = ir_builder->CreateBitCast(op0, data_ptr_type);
  CreateCall(&callee, {ret_buf_ptr, op0, ir_builder->getInt64(lhs_elms)});
  ir_mapping_[*inst] = ret_buf;
}

} // namespace halo