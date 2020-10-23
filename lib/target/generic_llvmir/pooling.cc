//===- pooling.cc ---------------------------------------------------------===//
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

void GenericLLVMIRCodeGen::RunOnInstruction(PoolingMaxInst* inst0) {
  PoolingMaxInst& inst = *inst0;
  llvm::IRBuilder<>* ir_builder = current_llvm_builder_;
  const Def& lhs = inst.GetOperand(0);
  llvm::Value* op0 = ir_mapping_[lhs];

  std::string fname =
      GetRTLibFuncName(inst, lhs.GetType().GetDataType(), inst.GetDataFormat());
  llvm::Type* ptr_ty =
      SNTypeToLLVMType(lhs.GetType().GetDataType())->getPointerTo();

  llvm::Type* op0_ty = op0->getType();
  if (!op0_ty->isPointerTy()) {
    auto buf =
        ir_builder->CreateAlloca(TensorTypeToLLVMType(lhs.GetType(), false),
                                 nullptr, lhs.GetOwner()->GetName() + "_buf");
    ir_builder->CreateStore(op0, buf);
    op0 = buf;
  }
  llvm::Type* int64_ty = ir_builder->getInt64Ty();
  llvm::FunctionType* ftype = llvm::FunctionType::get(
      ir_builder->getVoidTy(),
      {ptr_ty, ptr_ty, int64_ty, int64_ty, int64_ty, int64_ty, int64_ty,
       int64_ty, int64_ty, int64_ty, int64_ty, int64_ty, int64_ty, int64_ty,
       int64_ty, int64_ty},
      false);

  auto llvm_module = ir_builder->GetInsertBlock()->getParent()->getParent();
  llvm::FunctionCallee callee = llvm_module->getOrInsertFunction(fname, ftype);

  HLCHECK((inst.GetDataFormat() == DataFormat::NHWC ||
           inst.GetDataFormat() == DataFormat::NCHW) &&
          "Data format NHWC or NCHW is expected");

  const auto& info = ImageAxisInfo::GetImageAxisInfo(
      inst.GetDataFormat(), inst.GetDataFormat() /* filter format */);

  llvm::Value* data = ir_builder->CreateBitCast(op0, ptr_ty);
  llvm::Value* batch = ir_builder->getInt64(
      lhs.GetType().GetNumOfElementsInDim(info.batch_axis));
  llvm::Value* spatial_h = ir_builder->getInt64(
      lhs.GetType().GetNumOfElementsInDim(info.data_height_axis));
  llvm::Value* spatial_w = ir_builder->getInt64(
      lhs.GetType().GetNumOfElementsInDim(info.data_width_axis));
  llvm::Value* channel = ir_builder->getInt64(
      lhs.GetType().GetNumOfElementsInDim(info.data_channel_axis));
  llvm::Value* kernel_h =
      ir_builder->getInt64(inst.GetKsize()[info.kernel_height_axis]);
  llvm::Value* kernel_w =
      ir_builder->getInt64(inst.GetKsize()[info.kernel_width_axis]);
  llvm::Value* output_h = ir_builder->getInt64(
      inst.GetResultType().GetNumOfElementsInDim(info.data_height_axis));
  llvm::Value* output_w = ir_builder->getInt64(
      inst.GetResultType().GetNumOfElementsInDim(info.data_width_axis));
  llvm::Value* stride_h =
      ir_builder->getInt64(inst.GetStrides()[info.data_height_axis]);
  llvm::Value* stride_w =
      ir_builder->getInt64(inst.GetStrides()[info.data_width_axis]);
  llvm::Value* padding_left = ir_builder->getInt64(inst.GetPaddingLeft());
  llvm::Value* padding_right = ir_builder->getInt64(inst.GetPaddingRight());
  llvm::Value* padding_top = ir_builder->getInt64(inst.GetPaddingTop());
  llvm::Value* padding_bottom = ir_builder->getInt64(inst.GetPaddingBottom());

  llvm::Value* result = AllocateLLVMBuffer(ir_builder, Def{&inst, 0});

  llvm::Value* ret_buf_ptr = ir_builder->CreateBitCast(result, ptr_ty);
  CreateCall(&callee,
             {ret_buf_ptr, data, batch, spatial_h, spatial_w, channel, output_h,
              output_w, kernel_h, kernel_w, stride_h, stride_w, padding_left,
              padding_right, padding_top, padding_bottom});
  ir_mapping_[inst] = result;
}

} // namespace halo