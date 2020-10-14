//===- transpose.cc -------------------------------------------------------===//
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

void GenericLLVMIRCodeGen::RunOnInstruction(TransposeInst* inst) {
  llvm::IRBuilder<>* ir_builder = current_llvm_builder_;
  const Def& params = inst->GetOperand(0);
  llvm::Value* op0 = ir_mapping_[params];

  std::string fname = GetRTLibFuncName(*inst, params.GetType().GetDataType());

  // llvm::Type* op0_ty = op0->getType();
#if 0
  if (!op0_ty->isPointerTy()) {
    auto buf = ir_builder->CreateAlloca(
        TensorTypeToLLVMType(params.GetType(), false), nullptr,
        params.GetOwner()->GetName() + "_buf");
    ir_builder->CreateStore(op0, buf);
    op0 = buf;
  }
#endif

  llvm::Type* data_ptr_type =
      SNTypeToLLVMType(params.GetType().GetDataType())->getPointerTo();
  llvm::Type* int64_ptr_ty = ir_builder->getInt64Ty()->getPointerTo();
  llvm::Type* int32_ty = ir_builder->getInt32Ty();
  llvm::Type* int32_ptr_ty = int32_ty->getPointerTo();
  llvm::FunctionType* ftype = llvm::FunctionType::get(
      ir_builder->getVoidTy(),
      {data_ptr_type, data_ptr_type, int32_ptr_ty, int64_ptr_ty, int32_ty},
      false);

  auto llvm_module = ir_builder->GetInsertBlock()->getParent()->getParent();
  llvm::FunctionCallee callee = llvm_module->getOrInsertFunction(fname, ftype);

  auto dim = params.GetType().GetNumOfDims();
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

  const std::vector<int>& perm = inst->GetPermutation();
  HLCHECK(!perm.empty() && "constant perm is expected.");
  HLCHECK(perm.size() == dim);
  llvm::Type* perm_type = llvm::ArrayType::get(ir_builder->getInt32Ty(), dim);
  llvm::ArrayRef<int> perm_data(perm);
  llvm::Constant* perm_cv =
      llvm::ConstantDataArray::get(llvm_module_->getContext(), perm_data);
  auto perm_gv = llvm_module_->getOrInsertGlobal(
      params.GetOwner()->GetName() + "_perm", perm_type);
  gv = llvm::dyn_cast<llvm::GlobalVariable>(perm_gv);
  HLCHECK(gv != nullptr);
  gv->setInitializer(perm_cv);
  gv->setLinkage(llvm::GlobalValue::LinkageTypes::InternalLinkage);

  llvm::Value* perm_gv_ptr = ir_builder->CreateBitCast(gv, int32_ptr_ty);

  llvm::Value* input0 = ir_builder->CreateBitCast(op0, data_ptr_type);

  llvm::Value* dim_v = ir_builder->getInt32(dim);
  llvm::Value* ret_buf = AllocateLLVMBuffer(ir_builder, Def{inst, 0});

  llvm::Value* ret_buf_ptr = ir_builder->CreateBitCast(ret_buf, data_ptr_type);

  CreateCall(&callee, {ret_buf_ptr, input0, perm_gv_ptr, shape_gv_ptr, dim_v});

  ir_mapping_[*inst] = ret_buf;
}

} // end namespace halo
