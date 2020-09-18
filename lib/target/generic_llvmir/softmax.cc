//===- softmax.cc ---------------------------------------------------------===//
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
#include "llvm/IR/IRBuilder.h"

namespace halo {

void GenericLLVMIRCodeGen::RunOnInstruction(SoftmaxInst* inst) {
  llvm::IRBuilder<>* ir_builder = current_llvm_builder_;
  const Def& lhs = inst->GetOperand(0);
  const auto& lhs_type = lhs.GetType();
  DataType lhs_datatype = lhs_type.GetDataType();
  llvm::Value* op0 = ir_mapping_[lhs];

  std::string fname = GetRTLibFuncName(*inst, lhs_datatype);

  auto llvm_module = ir_builder->GetInsertBlock()->getParent()->getParent();
  llvm::PointerType* data_ptr_type =
      SNTypeToLLVMType(lhs_datatype)->getPointerTo();
  llvm::Type* int32_type = ir_builder->getInt32Ty();
  llvm::Type* int64_type = ir_builder->getInt64Ty();
  llvm::Type* int64_ptr_type = int64_type->getPointerTo();

  llvm::FunctionType* ftype =
      llvm::FunctionType::get(ir_builder->getVoidTy(),
                              {data_ptr_type, data_ptr_type, int64_ptr_type,
                               int32_type, int32_type, int64_type},
                              false);

  llvm::FunctionCallee callee = llvm_module->getOrInsertFunction(fname, ftype);

  if (!op0->getType()->isPointerTy()) {
    auto buf =
        ir_builder->CreateAlloca(TensorTypeToLLVMType(lhs_type, false), nullptr,
                                 lhs.GetOwner()->GetName() + "_buf");
    ir_builder->CreateStore(op0, buf);
    op0 = buf;
  }

  llvm::Value* logistic = ir_builder->CreateBitCast(op0, data_ptr_type);
  auto dim = lhs_type.GetNumOfDims();
  auto num_of_elements = lhs_type.GetTotalNumOfElements();
  llvm::Value* noe_v = ir_builder->getInt64(num_of_elements);
  llvm::Value* dim_v = ir_builder->getInt32(dim);
  // axis range is [-dim, dim-1]
  auto axis = inst->GetAxis();
  if (axis < 0) {
    axis = dim + axis;
  }
  llvm::Value* axis_v = ir_builder->getInt32(axis);
  llvm::Type* shape_type = llvm::ArrayType::get(int64_type, dim);
  llvm::ArrayRef<int64_t> shape_data(lhs_type.GetDimSizes());
  llvm::Constant* shape_cv =
      llvm::ConstantDataArray::get(llvm_module_->getContext(), shape_data);
  auto shape_gv = llvm_module_->getOrInsertGlobal(
      lhs.GetOwner()->GetName() + "_shape", shape_type);
  llvm::GlobalVariable* gv = llvm_module_->getNamedGlobal(shape_gv->getName());
  gv->setInitializer(shape_cv);
  gv->setLinkage(llvm::GlobalValue::LinkageTypes::InternalLinkage);

  llvm::Value* result = AllocateLLVMBuffer(ir_builder, Def{inst, 0});
  llvm::Value* ret_buf_ptr = ir_builder->CreateBitCast(result, data_ptr_type);
  llvm::Value* shape_gv_ptr = ir_builder->CreateBitCast(gv, int64_ptr_type);

  CreateCall(&callee,
             {ret_buf_ptr, logistic, shape_gv_ptr, axis_v, dim_v, noe_v});

  ir_mapping_[*inst] = result;
}

} // namespace halo