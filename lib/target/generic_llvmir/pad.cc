//===- pad.cc -------------------------------------------------------------===//
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
#include <string>

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/IRBuilder.h"
#include "halo/lib/ir/common_reduction_instructions.h"
#include "halo/lib/ir/ir_builder.h"
#include "halo/lib/ir/nn_cnn_instructions.h"
#include "halo/lib/ir/nn_instructions.h"
#include "halo/lib/target/generic_llvmir/generic_llvmir_codegen.h"

namespace halo {

void GenericLLVMIRCodeGen::RunOnInstruction(PadInst* inst) {
  llvm::IRBuilder<>* ir_builder = current_llvm_builder_;
  const Def& lhs = inst->GetOperand(0);
  const Def& rhs = inst->GetOperand(1);

  llvm::Value* op0 = ir_mapping_[lhs];
  llvm::Value* op1 = ir_mapping_[rhs];

  std::string fname = GetRTLibFuncName(*inst, lhs.GetType().GetDataType());

  auto llvm_module = ir_builder->GetInsertBlock()->getParent()->getParent();
  llvm::PointerType* data_ptr_type =
      SNTypeToLLVMType(lhs.GetType().GetDataType())->getPointerTo();
  llvm::Type* i32_type = ir_builder->getInt32Ty();
  llvm::Type* i32_ptr_type = i32_type->getPointerTo();
  llvm::Type* i64_ptr_type = ir_builder->getInt64Ty()->getPointerTo();
  llvm::FunctionType* ftype = llvm::FunctionType::get(
      ir_builder->getVoidTy(),
      {data_ptr_type, data_ptr_type, i32_type, i64_ptr_type, i32_ptr_type},
      false);

  llvm::FunctionCallee callee = llvm_module->getOrInsertFunction(fname, ftype);

  if (!op0->getType()->isPointerTy()) {
    auto buf =
        ir_builder->CreateAlloca(TensorTypeToLLVMType(lhs.GetType(), false),
                                 nullptr, lhs.GetOwner()->GetName() + "_buf");
    ir_builder->CreateStore(op0, buf);
    op0 = buf;
  }
  llvm::Value* input = ir_builder->CreateBitCast(op0, data_ptr_type);

  llvm::Value* ret_buf = AllocateLLVMBuffer(ir_builder, Def{inst, 0});

  auto& lhs_type = lhs.GetType();
  llvm::Value* dims = ir_builder->getInt32(lhs_type.GetNumOfDims());

  llvm::Value* ret_buf_ptr = ir_builder->CreateBitCast(ret_buf, data_ptr_type);

  const llvm::ArrayRef<int64_t> orig_shape(lhs_type.GetDimSizes());
  llvm::Constant* orig_shape_data =
      llvm::ConstantDataArray::get(llvm_module->getContext(), orig_shape);
  auto v = llvm_module_->getOrInsertGlobal(
      "shape_of_" + std::to_string(lhs.GetOwner()->GetId()),
      orig_shape_data->getType());
  llvm::GlobalVariable* gv = llvm_module_->getNamedGlobal(v->getName());
  gv->setInitializer(orig_shape_data);
  gv->setLinkage(llvm::GlobalValue::LinkageTypes::InternalLinkage);

  op1 = ir_builder->CreateBitCast(op1, i32_ptr_type);

  CreateCall(&callee, {ret_buf_ptr, input, dims,
                       ir_builder->CreateBitCast(gv, i64_ptr_type), op1});

  ir_mapping_[*inst] = ret_buf;
}

} // namespace halo