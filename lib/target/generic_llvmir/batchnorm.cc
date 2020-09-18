//===- batchnorm.cc -------------------------------------------------------===//
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

void GenericLLVMIRCodeGen::RunOnInstruction(BatchNormInst* inst) {
  llvm::IRBuilder<>* ir_builder = current_llvm_builder_;
  const Def& data = inst->GetOperand(0);
  std::string fname = GetRTLibFuncName(*inst, data.GetType().GetDataType(),
                                       inst->GetDataFormat());
  llvm::Type* float32_ty = ir_builder->getFloatTy();
  llvm::Type* ptr_ty = float32_ty->getPointerTo();
  llvm::Type* int64_ty = ir_builder->getInt64Ty();
  llvm::Type* int1_ty = ir_builder->getInt1Ty();
  llvm::FunctionType* ftype = llvm::FunctionType::get(
      ir_builder->getVoidTy(),
      {ptr_ty, ptr_ty, ptr_ty, ptr_ty, ptr_ty, ptr_ty, int64_ty, int64_ty,
       int64_ty, int64_ty, int1_ty, int1_ty, float32_ty, float32_ty},
      false);

  auto llvm_module = ir_builder->GetInsertBlock()->getParent()->getParent();
  llvm::FunctionCallee callee = llvm_module->getOrInsertFunction(fname, ftype);

  HLCHECK(inst->GetDataFormat() == DataFormat::NHWC ||
          inst->GetDataFormat() == DataFormat::NCHW);
  llvm::SmallVector<llvm::Value*, 5> ops; // NOLINT.
  for (size_t i = 0; i < inst->GetNumOfOperands(); ++i) {
    const Def& def = inst->GetOperand(i);
    llvm::Value* op_i = ir_mapping_[def];
    HLCHECK(op_i != nullptr);
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
  bool scalar_offset{false};
  bool scalar_scale{false};
  if (ops.size() <= 3) {
    scalar_offset = true;
    auto buf = ir_builder->CreateAlloca(float32_ty, nullptr, "offset_buf");
    llvm::Value* offset_v =
        llvm::ConstantFP::get(float32_ty, inst->GetOffset());
    ir_builder->CreateStore(offset_v, buf);
    ops.push_back(buf);
  }
  if (ops.size() <= 4) {
    scalar_scale = true;
    auto buf = ir_builder->CreateAlloca(float32_ty, nullptr, "scale_buf");
    llvm::Value* scale_v = llvm::ConstantFP::get(float32_ty, inst->GetScale());
    ir_builder->CreateStore(scale_v, buf);
    ops.push_back(buf);
  }
  llvm::Value* epsilon = llvm::ConstantFP::get(float32_ty, inst->GetEpsilon());
  llvm::Value* pre_scale =
      llvm::ConstantFP::get(float32_ty, inst->GetPreScalingFactor());

  const auto& info = ImageAxisInfo::GetImageAxisInfo(
      inst->GetDataFormat(), DataFormat::INVALID /* filter format */);
  llvm::Value* batch = ir_builder->getInt64(
      data.GetType().GetNumOfElementsInDim(info.batch_axis));
  llvm::Value* spatial_h = ir_builder->getInt64(
      data.GetType().GetNumOfElementsInDim(info.data_height_axis));
  llvm::Value* spatial_w = ir_builder->getInt64(
      data.GetType().GetNumOfElementsInDim(info.data_width_axis));
  llvm::Value* channel = ir_builder->getInt64(
      data.GetType().GetNumOfElementsInDim(info.data_channel_axis));

  llvm::Value* result = AllocateLLVMBuffer(ir_builder, Def{inst, 0});

  llvm::Value* ret_buf_ptr = ir_builder->CreateBitCast(result, ptr_ty);
  CreateCall(&callee,
             /* TODO(unknown): we need to do fixup earlier */
             {ret_buf_ptr, ops[0], ops[3], ops[4], ops[2], ops[1], batch,
              spatial_h, spatial_w, channel, ir_builder->getInt1(scalar_offset),
              ir_builder->getInt1(scalar_scale), epsilon, pre_scale});

  ir_mapping_[*inst] = result;
}

} // end namespace halo
