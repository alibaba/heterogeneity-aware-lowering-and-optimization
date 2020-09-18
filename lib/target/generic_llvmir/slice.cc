//===- slice.cc -----------------------------------------------------------===//
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
#include "halo/lib/transforms/type_legalizer.h"

namespace halo {

void GenericLLVMIRCodeGen::RunOnInstruction(SliceInst* inst) {
  llvm::IRBuilder<>* ir_builder = current_llvm_builder_;
  const Def& params = inst->GetOperand(0);
  const Def& begin = inst->GetOperand(1);

  std::string fname = GetRTLibFuncName(*inst, params.GetType().GetDataType());

  llvm::SmallVector<llvm::Value*, 4> ops; // NOLINT.
  for (size_t i = 0; i < inst->GetNumOfOperands(); ++i) {
    const Def& def = inst->GetOperand(i);
    llvm::Value* op_i = ir_mapping_[def];
    if (!op_i->getType()->isPointerTy()) {
      auto buf =
          ir_builder->CreateAlloca(TensorTypeToLLVMType(def.GetType(), false),
                                   nullptr, def.GetOwner()->GetName() + "_buf");
      ir_builder->CreateStore(op_i, buf);
      op_i = buf;
    }
    llvm::PointerType* data_ptr_type =
        SNTypeToLLVMType(def.GetType().GetDataType())->getPointerTo();
    op_i = ir_builder->CreateBitCast(op_i, data_ptr_type);
    ops.push_back(op_i);
  }

  llvm::Type* data_ptr_type =
      SNTypeToLLVMType(params.GetType().GetDataType())->getPointerTo();
  llvm::Type* index_ptr_type =
      SNTypeToLLVMType(begin.GetType().GetDataType())->getPointerTo();
  llvm::Type* int32_ty = ir_builder->getInt32Ty();
  llvm::Type* int64_ptr_ty = ir_builder->getInt64Ty()->getPointerTo();
  llvm::FunctionType* ftype =
      llvm::FunctionType::get(ir_builder->getVoidTy(),
                              {data_ptr_type, data_ptr_type, index_ptr_type,
                               index_ptr_type, int64_ptr_ty, int32_ty},
                              false);

  auto llvm_module = ir_builder->GetInsertBlock()->getParent()->getParent();
  llvm::FunctionCallee callee = llvm_module->getOrInsertFunction(fname, ftype);

  llvm::Value* input0 = ir_builder->CreateBitCast(ops[0], data_ptr_type);
  llvm::Value* input1 = ir_builder->CreateBitCast(ops[1], index_ptr_type);
  llvm::Value* input2 = ir_builder->CreateBitCast(ops[2], index_ptr_type);

  auto dim = params.GetType().GetNumOfDims();
  llvm::Value* dim_v = ir_builder->getInt32(dim);
  llvm::Type* shape_type = llvm::ArrayType::get(ir_builder->getInt64Ty(), dim);
  llvm::ArrayRef<int64_t> shape_data(params.GetType().GetDimSizes());
  llvm::Constant* shape_cv =
      llvm::ConstantDataArray::get(llvm_module_->getContext(), shape_data);
  auto shape_gv = llvm_module_->getOrInsertGlobal(
      params.GetOwner()->GetName() + "_shape", shape_type);
  llvm::GlobalVariable* gv = llvm_module_->getNamedGlobal(shape_gv->getName());
  gv->setInitializer(shape_cv);
  gv->setLinkage(llvm::GlobalValue::LinkageTypes::InternalLinkage);

  llvm::Value* shape_gv_ptr = ir_builder->CreateBitCast(gv, int64_ptr_ty);

  llvm::Value* ret_buf = AllocateLLVMBuffer(ir_builder, Def{inst, 0});
  llvm::Value* ret_buf_ptr = ir_builder->CreateBitCast(ret_buf, data_ptr_type);

  CreateCall(&callee,
             {ret_buf_ptr, input0, input1, input2, shape_gv_ptr, dim_v});

  ir_mapping_[*inst] = ret_buf;
}

} // end namespace halo