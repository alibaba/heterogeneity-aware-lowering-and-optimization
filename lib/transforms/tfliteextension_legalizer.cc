//===- tfliteextension_legalizer.cc ---------------------------------------===//
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

#include "halo/lib/transforms/tfliteextension_legalizer.h"

#include <numeric>

#include "halo/api/halo_data.h"
#include "halo/lib/framework/common.h"
#include "halo/lib/ir/common_instructions.h"
#include "halo/lib/ir/extension_instructions.h"
#include "halo/lib/ir/ir_builder.h"
#include "halo/lib/transforms/transforms_util.h"

namespace halo {

static std::vector<Def> TryConvertExtraOperand(const TFLITEExtensionInst* ext,
                                               size_t operand_num,
                                               const ActivationType& act_type,
                                               IRObject* new_inst,
                                               IRBuilder* builder) {
  if (operand_num == 3) {
    auto bias = ext->GetOperand(2);
    const auto& orig_bias = DynCast<Constant>(bias.GetOwner());
    // do broadcast, need to be offline for performance
    auto shape = bias.GetType().GetDimSizes();
    HLCHECK(shape.size() == 1);
    const static int dims = 4;
    // conv2D:NHWC, broadcast on C
    std::vector<int64_t> new_shape(dims, 1);
    new_shape.back() = shape.back();
    ConstantBuilder cb(ext->GetParent()->GetParent());
    Constant* c = cb.CreateConstant(ext->GetName() + "_bias",
                                    Type{DataType::FLOAT32, shape},
                                    orig_bias->GetRawDataPtr());
    new_inst = builder->CreateAdd(ext->GetName() + "_add", *new_inst, *c);
  }

  switch (act_type) {
    case ActivationType::RELU: {
      new_inst = builder->CreateRelu(ext->GetName() + "_relu", *new_inst);
      break;
    }
    case ActivationType::RELU6: {
      new_inst = builder->CreateRelu6(ext->GetName() + "_relu", *new_inst);
      break;
    }
    case ActivationType::TANH: {
      new_inst = builder->CreateTanh(ext->GetName() + "_relu", *new_inst);
      break;
    }
    case ActivationType::NONE: {
      break;
    }
    default:
      HLCHECK(0 && "Unsupportted activation type");
      break;
  }
  return {*new_inst};
}

static std::vector<Def> ConvertADD(const TFLITEExtensionInst* ext,
                                   IRBuilder* builder) {
  ActivationType act_type = halo::ActivationType::NONE;
  IRObject* new_inst = builder->CreateAdd(
      ext->GetName() + "_add", ext->GetOperand(0), ext->GetOperand(1));
  return TryConvertExtraOperand(ext, ext->GetNumOfOperands(), act_type,
                                new_inst, builder);
}

static std::vector<Def> ConvertAvgPool2D(const TFLITEExtensionInst* ext,
                                         IRBuilder* builder) {
  std::vector<int> ksize(4, 1);
  ksize[1] = ext->GetAttributes()[0].get()->GetValueAsInteger();
  ksize[2] = ext->GetAttributes()[1].get()->GetValueAsInteger();

  ActivationType act_type =
      ext->GetAttributes()[2].get()->GetValueAsEnumActivationType();

  Padding padding = Padding::VALID;
  padding = ext->GetAttributes()[3].get()->GetValueAsEnumPadding();

  std::vector<int> strides(4, 1);
  strides[1] = ext->GetAttributes()[4].get()->GetValueAsInteger(); // NOLINT.
  strides[2] = ext->GetAttributes()[5].get()->GetValueAsInteger(); // NOLINT.

  HLCHECK(ext->GetNumOfOperands() == 1);
  auto input = ext->GetOperand(0);
  builder->SetInsertAfter(ext);

  IRObject* new_inst =
      builder->CreatePoolingAvg(ext->GetName() + "_avgpool", input);
  auto inst = Downcast<PoolingAvgInst>(new_inst);
  inst->SetKsize(ksize);
  inst->SetPadding(padding);
  inst->SetStrides(strides);
  inst->SetDataFormat(DataFormat::NHWC);

  return TryConvertExtraOperand(ext, ext->GetNumOfOperands(), act_type,
                                new_inst, builder);
}

static std::vector<Def> ConvertConv2D(const TFLITEExtensionInst* ext,
                                      IRBuilder* builder) {
  std::vector<int> dilations(4, 1);
  dilations[1] = ext->GetAttributes()[0].get()->GetValueAsInteger();
  dilations[2] = ext->GetAttributes()[1].get()->GetValueAsInteger();

  ActivationType act_type =
      ext->GetAttributes()[2].get()->GetValueAsEnumActivationType();

  Padding padding = Padding::VALID;
  padding = ext->GetAttributes()[3].get()->GetValueAsEnumPadding();

  std::vector<int> strides(4, 1);
  strides[1] = ext->GetAttributes()[4].get()->GetValueAsInteger(); // NOLINT.
  strides[2] = ext->GetAttributes()[5].get()->GetValueAsInteger(); // NOLINT.

  HLCHECK(ext->GetNumOfOperands() == 2 || ext->GetNumOfOperands() == 3);
  auto input = ext->GetOperand(0);
  auto weight = ext->GetOperand(1);
  builder->SetInsertAfter(ext);

  IRObject* new_inst =
      builder->CreateConv2D(ext->GetName() + "_conv", input, weight);
  Conv2DInst* inst = Downcast<Conv2DInst>(new_inst);
  inst->SetDilations(dilations);
  inst->SetPadding(padding);
  inst->SetStrides(strides);
  inst->SetDataFormat(DataFormat::NHWC);
  inst->SetFilterFormat(DataFormat::HWCN); // because of ConvertOHWI2HWIO

  return TryConvertExtraOperand(ext, ext->GetNumOfOperands(), act_type,
                                new_inst, builder);
}

static std::vector<Def> ConvertDepthwiseConv2D(const TFLITEExtensionInst* ext,
                                               IRBuilder* builder) {
  int multiplier = 1;
  std::vector<int> dilations(4, 1);
  for (const auto& attr : ext->GetAttributes()) {
    if (attr->GetName() == "depth_multiplier") {
      multiplier = attr->GetValueAsInteger();
    }
  }
  dilations[1] = ext->GetAttributes()[1].get()->GetValueAsInteger();
  dilations[2] = ext->GetAttributes()[2].get()->GetValueAsInteger();
  ActivationType act_type =
      ext->GetAttributes()[3].get()->GetValueAsEnumActivationType();

  Padding padding = Padding::VALID;
  padding = ext->GetAttributes()[4].get()->GetValueAsEnumPadding();

  std::vector<int> strides(4, 1);
  strides[1] = ext->GetAttributes()[5].get()->GetValueAsInteger(); // NOLINT.
  strides[2] = ext->GetAttributes()[6].get()->GetValueAsInteger(); // NOLINT.

  HLCHECK(ext->GetNumOfOperands() == 2 || ext->GetNumOfOperands() == 3);
  auto input = ext->GetOperand(0);
  auto weight = ext->GetOperand(1);
  auto weight_shape = weight.GetType().GetDimSizes();

  builder->SetInsertAfter(ext);

  IRObject* new_inst =
      builder->CreateConv2D(ext->GetName() + "_depthwise_conv", input, weight);
  Conv2DInst* inst = Downcast<Conv2DInst>(new_inst);
  inst->SetDilations(dilations);
  inst->SetPadding(padding);
  inst->SetStrides(strides);
  inst->SetDataFormat(DataFormat::NHWC);
  inst->SetFilterFormat(DataFormat::HWCN); // because of ConvertOHWI2HWIO
  // TFL uses 1HWO (parser changes it to HW1O).
  inst->SetGroup(static_cast<int>(weight_shape.back() / multiplier));
  return TryConvertExtraOperand(ext, ext->GetNumOfOperands(), act_type,
                                new_inst, builder);
}

static std::vector<Def> ConvertSqueeze(const TFLITEExtensionInst* ext,
                                       IRBuilder* builder) {
  return ConvertSqueezeImpl<TFLITEExtensionInst>(ext, builder, "squeeze_dims");
}

static std::vector<Def> ConvertTFLITEExtension(
    const TFLITEExtensionInst* tflite_inst, IRBuilder* builder) {
  switch (tflite_inst->GetExtOpCode()) {
    case TFLITEExtOpCode::AVERAGE_POOL_2D: {
      return ConvertAvgPool2D(tflite_inst, builder);
    }
    case TFLITEExtOpCode::CONV_2D: {
      return ConvertConv2D(tflite_inst, builder);
    }
    case TFLITEExtOpCode::DEPTHWISE_CONV_2D: {
      return ConvertDepthwiseConv2D(tflite_inst, builder);
    }
    case TFLITEExtOpCode::ADD: {
      return ConvertADD(tflite_inst, builder);
    }
    case TFLITEExtOpCode::SQUEEZE: {
      return ConvertSqueeze(tflite_inst, builder);
    }
    default: {
      HLCHECK(0 && "Unhandled");
    }
  }
  return std::vector<Def>{};
}

bool TFLITEExtensionLegalizer::RunOnBasicBlock(BasicBlock* bb) {
  IRBuilder builder(bb);
  bool changed = false;
  changed |= AppendReturnInst(bb);
  for (auto& inst_t : *bb) {
    Instruction* inst = inst_t.get();
    if (inst->GetOpCode() == OpCode::EXTENSION) {
      ExtensionInst* ext_inst = Downcast<ExtensionInst>(inst);
      if (ext_inst->GetExtensionKind() ==
          ExtensionInst::ExtensionKind::kExtension_TFLITE) {
        TFLITEExtensionInst* tflite_inst =
            Downcast<TFLITEExtensionInst>(ext_inst);
        auto new_defs = ConvertTFLITEExtension(tflite_inst, &builder);
        if (!new_defs.empty()) {
          tflite_inst->ReplaceAllUsesWith(new_defs);
        }
      }
    }
  }
  return changed;
}

} // end namespace halo
