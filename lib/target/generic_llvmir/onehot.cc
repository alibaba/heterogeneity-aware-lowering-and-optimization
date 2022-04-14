//===- onehot.cc ----------------------------------------------------------===//
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

#include "halo/lib/target/generic_llvmir/generic_llvmir_codegen.h"
#include "halo/lib/transforms/type_legalizer.h"
#include "llvm/IR/IRBuilder.h"

namespace halo {

void GenericLLVMIRCodeGen::RunOnInstruction(OneHotInst* inst) {
  llvm::IRBuilder<>* ir_builder = current_llvm_builder_;
  const Def& indices = inst->GetOperand(0);
  const Def& depth = inst->GetOperand(1);
  const Def& off_value = inst->GetOperand(2);
  const Def& on_value = inst->GetOperand(3);
  llvm::Value* op0 = ir_mapping_[indices];
  llvm::Value* op1 = ir_mapping_[depth];
  llvm::Value* op2 = ir_mapping_[off_value];
  llvm::Value* op3 = ir_mapping_[on_value];

  std::string fname = GetRTLibFuncName(*inst, on_value.GetType().GetDataType());

  llvm::Type* data_type = SNTypeToLLVMType(on_value.GetType().GetDataType());
  llvm::Type* data_ptr_type = data_type->getPointerTo();
  llvm::Type* int64_ty = ir_builder->getInt64Ty();
  llvm::Type* int32_ptr_ty = ir_builder->getInt32Ty()->getPointerTo();
  llvm::Type* indices_ptr_ty =
      SNTypeToLLVMType(indices.GetType().GetDataType())->getPointerTo();
  llvm::FunctionType* ftype =
      llvm::FunctionType::get(ir_builder->getVoidTy(),
                              {data_ptr_type, indices_ptr_ty, int32_ptr_ty,
                               data_ptr_type, data_ptr_type, int64_ty},
                              false);

  auto llvm_module = ir_builder->GetInsertBlock()->getParent()->getParent();
  llvm::FunctionCallee callee = llvm_module->getOrInsertFunction(fname, ftype);

  // TODO(unknown): handle other axis
  HLCHECK(inst->GetAxis() == -1 && "unsupported axis value for onehot.");
  llvm::Type* op0_ty = op0->getType();
  if (!op0_ty->isPointerTy()) {
    auto buf = ir_builder->CreateAlloca(
        TensorTypeToLLVMType(indices.GetType(), false), nullptr,
        indices.GetOwner()->GetName() + "_buf");
    ir_builder->CreateStore(op0, buf);
    op0 = buf;
  }
  llvm::Value* input0 = ir_builder->CreateBitCast(op0, indices_ptr_ty);

  HLCHECK(op1->getType()->isPointerTy() && op2->getType()->isPointerTy() &&
          op3->getType()->isPointerTy());
  llvm::Value* input1 = ir_builder->CreateBitCast(op1, int32_ptr_ty);
  llvm::Value* input2 = ir_builder->CreateBitCast(op2, data_ptr_type);
  llvm::Value* input3 = ir_builder->CreateBitCast(op3, data_ptr_type);

  llvm::Value* noe_indicies =
      ir_builder->getInt64(indices.GetType().GetTotalNumOfElements());
  llvm::Value* ret_buf = AllocateLLVMBuffer(ir_builder, Def{inst, 0});
  llvm::Value* ret_buf_ptr = ir_builder->CreateBitCast(ret_buf, data_ptr_type);

  CreateCall(&callee,
             {ret_buf_ptr, input0, input1, input2, input3, noe_indicies});
  ir_mapping_[*inst] = ret_buf;
}

} // end namespace halo
