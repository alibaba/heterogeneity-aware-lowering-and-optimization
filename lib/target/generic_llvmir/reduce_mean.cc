//===- reduce_mean.cc -----------------------------------------------------===//
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

void GenericLLVMIRCodeGen::RunOnCommonReductionInstruction(
    Instruction* inst, const std::vector<int>& axis) {
  llvm::IRBuilder<>* ir_builder = current_llvm_builder_;
  const Def& lhs = inst->GetOperand(0);
  llvm::Value* op0 = ir_mapping_[lhs];
  auto& lhs_type = lhs.GetType();
  const halo::Type& result_type = inst->GetResultsTypes()[0];

  std::string fname = GetRTLibFuncName(*inst, lhs_type.GetDataType());

  auto llvm_module = ir_builder->GetInsertBlock()->getParent()->getParent();
  llvm::PointerType* ret_ptr_type =
      SNTypeToLLVMType(result_type.GetDataType())->getPointerTo();
  llvm::PointerType* data_ptr_type =
      SNTypeToLLVMType(lhs_type.GetDataType())->getPointerTo();
  llvm::Type* i32_type = ir_builder->getInt32Ty();
  llvm::Type* i64_type = ir_builder->getInt64Ty();
  llvm::Type* i32_ptr_type = i32_type->getPointerTo();
  llvm::Type* i64_ptr_type = i64_type->getPointerTo();
  llvm::FunctionType* ftype =
      llvm::FunctionType::get(ir_builder->getVoidTy(),
                              {ret_ptr_type, data_ptr_type, i64_ptr_type,
                               i32_ptr_type, i64_type, i32_type, i32_type},
                              false);

  llvm::FunctionCallee callee = llvm_module->getOrInsertFunction(fname, ftype);

  if (!op0->getType()->isPointerTy()) {
    auto buf =
        ir_builder->CreateAlloca(TensorTypeToLLVMType(lhs_type, false), nullptr,
                                 lhs.GetOwner()->GetName() + "_buf");
    ir_builder->CreateStore(op0, buf);
    op0 = buf;
  }
  llvm::Value* input = ir_builder->CreateBitCast(op0, data_ptr_type);

  auto dim = lhs_type.GetNumOfDims();
  auto num_of_elements = lhs_type.GetTotalNumOfElements();
  llvm::Value* noe_v = ir_builder->getInt64(num_of_elements);
  llvm::Value* dim_v = ir_builder->getInt32(dim);
  llvm::Type* shape_type = llvm::ArrayType::get(i64_type, dim);
  llvm::ArrayRef<int64_t> shape_data(lhs_type.GetDimSizes());
  llvm::Constant* shape_cv =
      llvm::ConstantDataArray::get(llvm_module_->getContext(), shape_data);
  auto shape_gv = llvm_module_->getOrInsertGlobal(
      lhs.GetOwner()->GetName() + "_shape", shape_type);
  llvm::GlobalVariable* gv = llvm_module_->getNamedGlobal(shape_gv->getName());
  gv->setInitializer(shape_cv);
  gv->setLinkage(llvm::GlobalValue::LinkageTypes::InternalLinkage);
  llvm::Value* shape_gv_ptr = ir_builder->CreateBitCast(gv, i64_ptr_type);

  int axis_size = 0;
  llvm::Value* axis_v = nullptr;
  if (inst->GetNumOfOperands() > 1) {
    const Def& rhs = inst->GetOperand(1);
    llvm::Value* op1 = ir_mapping_[rhs];
    auto& rhs_type = rhs.GetType();
    axis_size = rhs_type.GetTotalNumOfElements();
    if (!op1->getType()->isPointerTy()) {
      auto buf =
          ir_builder->CreateAlloca(TensorTypeToLLVMType(rhs_type, false),
                                   nullptr, rhs.GetOwner()->GetName() + "_buf");
      ir_builder->CreateStore(op1, buf);
      op1 = buf;
    }
    axis_v = ir_builder->CreateBitCast(op1, i32_ptr_type);
  } else {
    // TODO(unknown): handle axis[i] < 0
    axis_size = axis.size();
    HLCHECK(axis_size && "axis is expected to be non-empty");
    llvm::Type* axis_type = llvm::ArrayType::get(i32_type, axis_size);
    llvm::ArrayRef<int> axis_data(axis);
    llvm::Constant* axis_cv =
        llvm::ConstantDataArray::get(llvm_module_->getContext(), axis_data);
    auto axis_gv = llvm_module_->getOrInsertGlobal(
        inst->GetName() + "_reduction_axis", axis_type);
    llvm::GlobalVariable* gv = llvm_module_->getNamedGlobal(axis_gv->getName());
    gv->setInitializer(axis_cv);
    gv->setLinkage(llvm::GlobalValue::LinkageTypes::InternalLinkage);
    axis_v = ir_builder->CreateBitCast(gv, i32_ptr_type);
  }
  llvm::Value* axis_size_v = ir_builder->getInt32(axis_size);

  llvm::Value* ret_buf = AllocateLLVMBuffer(ir_builder, Def{inst, 0});

  llvm::Value* ret_buf_ptr = ir_builder->CreateBitCast(ret_buf, ret_ptr_type);

  CreateCall(&callee, {ret_buf_ptr, input, shape_gv_ptr, axis_v, noe_v, dim_v,
                       axis_size_v});

  ir_mapping_[*inst] = ret_buf;
}

} // namespace halo