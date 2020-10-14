//===- math_binary.cc -----------------------------------------------------===//
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
#include <vector>

#include "halo/lib/ir/ir_builder.h"
#include "halo/lib/target/generic_llvmir/generic_llvmir_codegen.h"
#include "halo/lib/transforms/type_legalizer.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/IRBuilder.h"

namespace halo {
void GenericLLVMIRCodeGen::RunOnMathBinaryInstruction(Instruction* inst) {
  llvm::IRBuilder<>* ir_builder = current_llvm_builder_;
  const Def& lhs = inst->GetOperand(0);
  const Def& rhs = inst->GetOperand(1);

  llvm::Value* op0 = ir_mapping_[lhs];
  llvm::Value* op1 = ir_mapping_[rhs];

  // TODO(unknown): we can split into multiple smaller vec ops.
  llvm::PointerType* data_ptr_type =
      SNTypeToLLVMType(lhs.GetType().GetDataType())->getPointerTo();
  llvm::Type* i64_type = ir_builder->getInt64Ty();
  llvm::PointerType* i64_ptr_type = i64_type->getPointerTo();
  llvm::Type* i32_type = ir_builder->getInt32Ty();
  llvm::Type* bool_type = ir_builder->getInt1Ty();
  llvm::FunctionType* ftype = llvm::FunctionType::get(
      ir_builder->getVoidTy(),
      {data_ptr_type, data_ptr_type, data_ptr_type, i64_type, bool_type,
       i64_ptr_type, i64_ptr_type, i64_ptr_type, i32_type},
      false);

  std::string fname = GetRTLibFuncName(*inst, lhs.GetType().GetDataType());

  llvm::FunctionCallee callee = llvm_module_->getOrInsertFunction(fname, ftype);

  llvm::Value* ret_buf = AllocateLLVMBuffer(ir_builder, Def{inst, 0});
  auto ret_buf_ptr = ir_builder->CreateBitCast(ret_buf, data_ptr_type);
  op0 = ir_builder->CreateBitCast(op0, data_ptr_type);
  op1 = ir_builder->CreateBitCast(op1, data_ptr_type);

  const halo::Type& ret_type = inst->GetResultsTypes()[0];
  const halo::Type& lhs_type = lhs.GetType();
  const halo::Type& rhs_type = rhs.GetType();
  int dim = ret_type.GetNumOfDims();
  auto noe = ret_type.GetTotalNumOfElements();
  llvm::Value* dim_v = ir_builder->getInt32(dim);
  llvm::Value* noe_v = ir_builder->getInt64(noe);

  if (lhs_type == rhs_type) {
    llvm::Value* null_ptr = llvm::ConstantPointerNull::get(i64_ptr_type);
    CreateCall(&callee,
               {ret_buf_ptr, op0, op1, noe_v, ir_builder->getInt1(false),
                null_ptr, null_ptr, null_ptr, dim_v});
  } else {
    llvm::Type* shape_type =
        llvm::ArrayType::get(ir_builder->getInt64Ty(), dim);
    llvm::ArrayRef<int64_t> shape_data(ret_type.GetDimSizes());
    llvm::Constant* shape_cv =
        llvm::ConstantDataArray::get(llvm_module_->getContext(), shape_data);
    auto shape_gv =
        llvm_module_->getOrInsertGlobal(inst->GetName() + "_shape", shape_type);
    llvm::GlobalVariable* gv =
        llvm_module_->getNamedGlobal(shape_gv->getName());
    gv->setInitializer(shape_cv);
    gv->setLinkage(llvm::GlobalValue::LinkageTypes::InternalLinkage);

    llvm::Value* shape_gv_ptr = ir_builder->CreateBitCast(gv, i64_ptr_type);

    std::vector<int64_t> lhs_dims(dim, 1);
    for (int i = dim - 1, j = static_cast<int>(lhs_type.GetNumOfDims()) - 1;
         j >= 0; i--, j--) {
      lhs_dims[i] = lhs_type.GetNumOfElementsInDim(j);
    }
    llvm::ArrayRef<int64_t> lhs_shape_data(lhs_dims);
    shape_cv = llvm::ConstantDataArray::get(llvm_module_->getContext(),
                                            lhs_shape_data);
    shape_gv = llvm_module_->getOrInsertGlobal(
        lhs.GetOwner()->GetName() + "_shape", shape_type);
    gv = llvm_module_->getNamedGlobal(shape_gv->getName());
    gv->setInitializer(shape_cv);
    gv->setLinkage(llvm::GlobalValue::LinkageTypes::InternalLinkage);

    llvm::Value* lhs_shape_gv_ptr = ir_builder->CreateBitCast(gv, i64_ptr_type);

    std::vector<int64_t> rhs_dims(dim, 1);
    for (int i = dim - 1, j = static_cast<int>(rhs_type.GetNumOfDims()) - 1;
         j >= 0; i--, j--) {
      rhs_dims[i] = rhs_type.GetNumOfElementsInDim(j);
    }
    llvm::ArrayRef<int64_t> rhs_shape_data(rhs_dims);
    shape_cv = llvm::ConstantDataArray::get(llvm_module_->getContext(),
                                            rhs_shape_data);
    shape_gv = llvm_module_->getOrInsertGlobal(
        rhs.GetOwner()->GetName() + "_shape", shape_type);
    gv = llvm_module_->getNamedGlobal(shape_gv->getName());
    gv->setInitializer(shape_cv);
    gv->setLinkage(llvm::GlobalValue::LinkageTypes::InternalLinkage);

    llvm::Value* rhs_shape_gv_ptr = ir_builder->CreateBitCast(gv, i64_ptr_type);

    CreateCall(&callee,
               {ret_buf_ptr, op0, op1, noe_v, ir_builder->getInt1(true),
                shape_gv_ptr, lhs_shape_gv_ptr, rhs_shape_gv_ptr, dim_v});
  }
  ir_mapping_[*inst] = ret_buf;
}

} // namespace halo