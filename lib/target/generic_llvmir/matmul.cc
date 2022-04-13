//===- matmul.cc ----------------------------------------------------------===//
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
#include "halo/lib/ir/math_instructions.h"
#include "halo/lib/target/generic_llvmir/generic_llvmir_codegen.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/IRBuilder.h"

namespace halo {

void GenericLLVMIRCodeGen::RunOnInstruction(MatMulInst* inst) {
  if (const auto ty = inst->GetResultType(); ty.GetNumOfDims() > 2) {
    BatchMatMulInst bm(inst->GetGlobalContext(), inst->GetName(),
                       inst->GetOperands());
    bm.SetTransposeA(inst->GetTransposeA());
    bm.SetTransposeB(inst->GetTransposeB());
    bm.GetResultsTypes()[0] = ty;
    RunOnInstruction(&bm);
    Def ret{&bm, 0};
    ir_mapping_[*inst] = ir_mapping_[ret];
    return;
  }

  llvm::IRBuilder<>* ir_builder = current_llvm_builder_;
  const Def& lhs = inst->GetOperand(0);
  const Def& rhs = inst->GetOperand(1);

  llvm::Value* op0 = ir_mapping_[lhs];
  llvm::Value* op1 = ir_mapping_[rhs];

  std::string fname = GetRTLibFuncName(*inst, lhs.GetType().GetDataType());

  auto llvm_module = ir_builder->GetInsertBlock()->getParent()->getParent();
  llvm::PointerType* ptr_type =
      SNTypeToLLVMType(lhs.GetType().GetDataType())->getPointerTo();
  //     llvm::dyn_cast<llvm::PointerType>(op0->getType());
  llvm::Type* bool_type = ir_builder->getInt1Ty();
  llvm::Type* int64_type = ir_builder->getInt64Ty();
  llvm::FunctionType* ftype = llvm::FunctionType::get(
      ir_builder->getVoidTy(),
      {ptr_type, ptr_type, ptr_type, int64_type, int64_type, int64_type,
       int64_type, bool_type, bool_type},
      false);

  llvm::FunctionCallee callee = llvm_module->getOrInsertFunction(fname, ftype);

  llvm::Value* param0 = ir_builder->CreateBitCast(op0, ptr_type);
  llvm::Value* param1 = ir_builder->CreateBitCast(op1, ptr_type);
  llvm::Value* dim_lhs_0 =
      ir_builder->getInt64(lhs.GetType().GetNumOfElementsInDim(0));
  llvm::Value* dim_lhs_1 =
      ir_builder->getInt64(lhs.GetType().GetNumOfElementsInDim(1));
  llvm::Value* dim_rhs_0 =
      ir_builder->getInt64(rhs.GetType().GetNumOfElementsInDim(0));
  llvm::Value* dim_rhs_1 =
      ir_builder->getInt64(rhs.GetType().GetNumOfElementsInDim(1));
  llvm::Value* transpose_a = ir_builder->getInt1(inst->GetTransposeA());
  llvm::Value* transpose_b = ir_builder->getInt1(inst->GetTransposeB());

  llvm::Value* ret_buf = ir_builder->CreateAlloca(
      TensorTypeToLLVMType(inst->GetResultType(), false), nullptr,
      inst->GetName());
  llvm::Value* ret_buf_ptr = ir_builder->CreateBitCast(ret_buf, ptr_type);
  CreateCall(&callee, {ret_buf_ptr, param0, param1, dim_lhs_0, dim_lhs_1,
                       dim_rhs_0, dim_rhs_1, transpose_a, transpose_b});
  ir_mapping_[*inst] = ret_buf;
}

} // namespace halo