//===- relu.cc ------------------------------------------------------------===//
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

#include "llvm/IR/IRBuilder.h"
#include "halo/lib/target/generic_llvmir/generic_llvmir_codegen.h"

namespace halo {

void GenericLLVMIRCodeGen::RunOnInstruction(ReluInst* inst) {
  llvm::IRBuilder<>* ir_builder = current_llvm_builder_;
  const Def& lhs = inst->GetOperand(0);
  llvm::Value* op0 = ir_mapping_[lhs];

  auto elems = inst->GetOperand(0).GetType().GetTotalNumOfElements();
  if (elems > GetMaxVectorSize()) {
    // TODO(unknown): we can split into multiple smaller vec ops.
    llvm::PointerType* data_ptr_type =
        SNTypeToLLVMType(lhs.GetType().GetDataType())->getPointerTo();
    llvm::Type* i64_type = ir_builder->getInt64Ty();

    llvm::FunctionType* ftype = llvm::FunctionType::get(
        ir_builder->getVoidTy(), {data_ptr_type, data_ptr_type, i64_type},
        false);

    std::string fname = GetRTLibFuncName(*inst, lhs.GetType().GetDataType());

    llvm::FunctionCallee callee =
        llvm_module_->getOrInsertFunction(fname, ftype);

    llvm::Value* ret_buf = AllocateLLVMBuffer(ir_builder, *inst);
    auto ret_buf_ptr = ir_builder->CreateBitCast(ret_buf, data_ptr_type);
    op0 = ir_builder->CreateBitCast(op0, data_ptr_type);
    CreateCall(&callee, {ret_buf_ptr, op0, ir_builder->getInt64(elems)});
    ir_mapping_[*inst] = ret_buf;
    return;
  }
  if (op0->getType()->isPointerTy()) {
    op0 = ir_builder->CreateLoad(op0);
  }

  llvm::Type* op_ty = op0->getType();
  HLCHECK(op_ty->isFPOrFPVectorTy() && "Floating point type is expected.");

  llvm::Constant* zero = llvm::Constant::getNullValue(op_ty);
  llvm::Value* result = ir_builder->CreateFCmpOGT(op0, zero);
  result = ir_builder->CreateSelect(result, op0, zero);
  ir_mapping_[*inst] = result;
}

} // namespace halo