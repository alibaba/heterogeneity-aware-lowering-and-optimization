//===- caffeextension_legalizer.cc ----------------------------------------===//
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

#include "halo/lib/transforms/caffeextension_legalizer.h"

#include <cmath>

#include "halo/api/halo_data.h"
#include "halo/lib/framework/common.h"
#include "halo/lib/ir/attribute.h"
#include "halo/lib/ir/common_instructions.h"
#include "halo/lib/ir/extension_instructions.h"
#include "halo/lib/ir/ir_builder.h"
#include "halo/lib/transforms/transforms_util.h"

namespace halo {

static std::vector<Def> ConvertConvolution(const CAFFEExtensionInst* ext,
                                           IRBuilder* builder) {
  std::vector<int> pad(2, 0);
  pad = FindAttributeValue(*ext, "pad", pad);
  if (pad.size() == 1) {
    pad.push_back(pad.front());
  }
  pad[0] = FindAttributeValue(*ext, "pad_h", pad[0]);
  pad[1] = FindAttributeValue(*ext, "pad_w", pad[1]);
  std::vector<int> stride(4, 1);
  stride = FindAttributeValue(*ext, "stride", stride);
  if (stride.size() == 1) {
    stride.push_back(stride.front());
  }
  while (stride.size() < 4) {
    stride.insert(stride.begin(), 1);
  }
  stride[2] = FindAttributeValue(*ext, "stride_h", stride[2]);
  stride[3] = FindAttributeValue(*ext, "stride_w", stride[3]);

  std::vector<int> dilation(4, 1);
  const auto value = FindAttributeValue(*ext, "dilation", dilation);
  // N and C set to 1
  dilation[2] = value.empty() ? 1 : value[0];
  dilation[3] = (value.size() > 1) ? value[1] : dilation[2];

  int group = FindAttributeValue(*ext, "group", 1);

  bool bias_term = FindAttributeValue(*ext, "bias_term", true);

  HLCHECK(ext->GetNumOfOperands() == 2 || ext->GetNumOfOperands() == 3);
  auto input = ext->GetOperand(0);
  auto weight = ext->GetOperand(1);

  IRObject* new_inst =
      builder->CreateConv2D(ext->GetName() + "_conv", input, weight);
  Conv2DInst* conv_inst = Downcast<Conv2DInst>(new_inst);
  conv_inst->SetPaddingLeft(pad[1]);
  conv_inst->SetPaddingRight(pad[1]);
  conv_inst->SetPaddingTop(pad[0]);
  conv_inst->SetPaddingBottom(pad[0]);
  conv_inst->SetStrides(stride);
  conv_inst->SetDilations(dilation);
  conv_inst->SetGroup(group);
  // conv_inst->SetKernelSize(kernel_size);
  // conv_inst->SetNumOutput(num_output);
  // conv_inst->SetBiasTerm(bias_term);
  conv_inst->SetPadding(Padding::EXPLICIT);
  conv_inst->SetDataFormat(DataFormat::NCHW);
  conv_inst->SetFilterFormat(DataFormat::NCHW);

  if (bias_term) {
    auto bias = ext->GetOperand(2);
    const auto& orig_bias = DynCast<Constant>(bias.GetOwner());
    // do broadcast, need to be offline for performance
    auto shape = bias.GetType().GetDimSizes();
    if (shape.size() > 1) {
      auto c = shape.back();
      shape.clear();
      shape.push_back(c);
    }
    const static int axis = 1; // broadcast on C
    const static int dims = 4; // conv2D
    HLCHECK(shape.size() == 1);
    if (shape.size() == 1) {
      for (int i = 0; i < axis; ++i) {
        shape.insert(shape.begin(), 1);
      }
      for (int i = 2; i < dims; ++i) {
        shape.emplace_back(1);
      }
    }
    ConstantBuilder cb(ext->GetParent()->GetParent());
    Constant* c = cb.CreateConstant(ext->GetName() + "_bias",
                                    Type{DataType::FLOAT32, shape},
                                    orig_bias->GetRawDataPtr());
    new_inst = builder->CreateAdd(ext->GetName() + "_add", *new_inst, *c);
  }
  return {*new_inst};
}

static std::vector<Def> ConvertDetectionOutput(const CAFFEExtensionInst* ext,
                                               IRBuilder* builder) {
  HLCHECK(ext->GetNumOfOperands() == 3);
  const auto& loc = ext->GetOperand(0);
  const auto& conf = ext->GetOperand(1);
  const auto& boxes = ext->GetOperand(2);

  if (!loc.GetType().IsValid() || !conf.GetType().IsValid() ||
      !boxes.GetType().IsValid()) {
    return {};
  }

  builder->SetInsertAfter(ext);
  auto inst = builder->CreateCustom(ext->GetName(), {loc, conf, boxes}, 1,
                                    "DetectionOutput");
  int top_k = FindAttributeValue(*ext, "keep_top_k", -1);
  int batch = loc.GetType().GetNumOfElementsInDim(0);
  constexpr int n = 7; // image_id, label, confidence, xmin, ymin, xmax, ymax
  inst->GetResultsTypes()[0] = Type{DataType::FLOAT32, {batch, top_k, n}};
  for (auto& attr : ext->GetAttributes()) {
    inst->AddOneAttribute(attr->Clone());
  }
  return {*inst};
}

static std::vector<Def> ConvertUpsample(const CAFFEExtensionInst* ext,
                                        IRBuilder* builder) {
  HLCHECK(ext->GetNumOfOperands() == 1);
  auto input = ext->GetOperand(0);
  if (!input.GetType().IsValid()) {
    return {};
  }
  int scale = 1;
  for (const auto& attr : ext->GetAttributes()) {
    if (attr->GetName() == "scale") {
      scale = attr->GetValueAsInteger();
    }
  }
  ConstantBuilder cb(ext->GetParent()->GetParent());
  auto rank = input.GetType().GetNumOfDims();
  HLCHECK(rank == 4);
  auto shape = input.GetType().GetDimSizes();
  shape[2] *= scale;
  shape[3] *= scale;
  auto scale_c = cb.CreateConstant(
      ext->GetName() + "_scale",
      Type{DataType::INT64, {static_cast<int64_t>(shape.size())}},
      shape.data());
  ResizeInst* resize = builder->CreateResize(ext->GetName(), input, *scale_c);
  resize->SetExplicitShape(true);
  return {*resize};
}

static std::vector<Def> ConvertEltwise(const CAFFEExtensionInst* ext,
                                       IRBuilder* builder) {
  HLCHECK(ext->GetNumOfOperands() == 2);
  auto lhs = ext->GetOperand(0);
  auto rhs = ext->GetOperand(1);

  std::unordered_map<std::string, int> attr_map;
  std::vector<OpCode> ops = {OpCode::MUL, OpCode::ADD, OpCode::MAXIMUM};
  OpCode op_code = ops[FindAttributeValue(*ext, "operation", 1)];

  if (HasAttribute(*ext, "coeff") && op_code == OpCode::ADD) {
    std::vector<float> coeff;
    coeff = FindAttributeValue(*ext, "coeff", coeff);
    if (coeff.size() == 2) {
      // TODO (unknown) support more case
      HLCHECK(coeff[0] == 1.0 && coeff[1] == -1.0);
      op_code = OpCode::SUB;
    }
  }
  auto new_inst = builder->CreateBinary(ext->GetName(), lhs, rhs, op_code);
  return {*new_inst};
}

static std::vector<Def> ConvertFlatten(const CAFFEExtensionInst* ext,
                                       IRBuilder* builder) {
  auto input = ext->GetOperand(0);
  const Type& input_type = input.GetType();
  if (!input_type.IsValid()) {
    return {};
  }

  HLCHECK(ext->GetNumOfAttributes() == 1);
  const Attribute* attr = ext->GetAttributes()[0].get();
  HLCHECK(attr->GetName() == "axis");
  int axis = attr->GetValueAsInteger();
  std::vector<int32_t> new_dims{1, 1};
  for (int i = 0, e = input_type.GetNumOfDims(); i < e; ++i) {
    if (i < axis) {
      new_dims[0] *= input_type.GetNumOfElementsInDim(i);
    } else {
      new_dims[1] *= input_type.GetNumOfElementsInDim(i);
    }
  }
  ConstantBuilder cb(ext->GetParent()->GetParent());
  Constant* c =
      cb.CreateConstant(ext->GetName() + "_flatten_dims",
                        Type{DataType::INT32, {2}, true}, new_dims.data());
  builder->SetInsertAfter(ext);
  auto new_inst = builder->CreateReshapeDynamic(ext->GetName(), {input, *c});
  return {*new_inst};
}

static std::vector<Def> ConvertNormalize(const CAFFEExtensionInst* ext,
                                         IRBuilder* builder) {
  HLCHECK(ext->GetNumOfOperands() == 2);
  const auto& input = ext->GetOperand(0);
  const auto& input_ty = input.GetType();

  auto scale = ext->GetOperand(1);
  const auto& scale_ty = scale.GetType();
  if (!input_ty.IsValid() || !scale_ty.IsValid()) {
    return {};
  }
  bool accross_spatial = FindAttributeValue<bool>(*ext, "across_spatial");
  bool channel_shared = FindAttributeValue<bool>(*ext, "channel_shared");
  float epsilon = FindAttributeValue<float>(*ext, "eps");
  std::vector<int> axes{1};
  if (accross_spatial) {
    for (int i = 2, e = input_ty.GetNumOfDims(); i < e; ++i) {
      axes.push_back(i);
    }
  }

  builder->SetInsertAfter(ext);

  if (!scale_ty.IsScalar() && (!channel_shared)) {
    HLCHECK(scale_ty.GetNumOfDims() == 1);
    HLCHECK(scale_ty.GetTotalNumOfElements() ==
            input_ty.GetNumOfElementsInDim(1));
    std::vector<int64_t> new_shape{scale_ty.GetNumOfElementsInDim(0), 1, 1};
    ConstantBuilder cb(ext->GetParent()->GetParent());
    Constant* c = cb.CreateConstant(
        ext->GetName() + "_shape",
        Type{DataType::INT64, {static_cast<int64_t>(new_shape.size())}, true},
        new_shape.data());
    scale = *builder->CreateReshapeDynamic(ext->GetName() + "_reshape",
                                           {scale, *c});
  }

  auto l2norm =
      builder->CreateLpNormalize(ext->GetName() + "_L2Norm", input, scale);
  l2norm->SetAxis(axes);
  l2norm->SetP(2);
  l2norm->SetEpsilon(epsilon);

  return {*l2norm};
}

static std::vector<Def> ConvertPool(const CAFFEExtensionInst* ext,
                                    IRBuilder* builder) {
  HLCHECK(ext->GetNumOfOperands() == 1);
  auto input = ext->GetOperand(0);
  const auto& input_type = input.GetType();
  bool global_pooling = FindAttributeValue<bool>(*ext, "global_pooling", false);
  int stride = FindAttributeValue(*ext, "stride", 1);
  int stride_h = FindAttributeValue(*ext, "stride_h", stride);
  int stride_w = FindAttributeValue(*ext, "stride_w", stride);
  int kernel_size = FindAttributeValue(*ext, "kernel_size", -1);
  int kernel_size_h = FindAttributeValue(*ext, "kernel_h", kernel_size);
  int kernel_size_w = FindAttributeValue(*ext, "kernel_w", kernel_size);
  int pool = FindAttributeValue(*ext, "pool", 0);
  HLCHECK(pool == 0 || pool == 1);

  int pad = FindAttributeValue(*ext, "pad", 0);
  int pad_h = FindAttributeValue(*ext, "pad_h", pad);
  int pad_w = FindAttributeValue(*ext, "pad_w", pad);

  bool ceil_mode = FindAttributeValue<bool>(*ext, "ceil_mode", true);
  // If ceil_mode is set to false (non-default), we always use it.
  int round_mode = !ceil_mode ? 1 : FindAttributeValue(*ext, "round_mode", 0);

  if (global_pooling) {
    if (!input_type.IsValid()) {
      return {};
    }
    kernel_size_h = input_type.GetNumOfElementsInDim(2);
    kernel_size_w = input_type.GetNumOfElementsInDim(3);

    stride_h = 1;
    stride_w = 1;

    pad_h = 0;
    pad_w = 0;
  }

  auto set_pooling_attributes = [&](auto inst) {
    inst->SetKsize({1, 1, kernel_size_h, kernel_size_w});
    inst->SetPaddingLeft(pad_w);
    inst->SetPaddingRight(pad_w);
    inst->SetPaddingTop(pad_h);
    inst->SetPaddingBottom(pad_h);
    inst->SetStrides({1, 1, stride_h, stride_w});
    inst->SetPadding(Padding::EXPLICIT);
    inst->SetDataFormat(DataFormat::NCHW);
    inst->SetRoundMode(round_mode == 0 ? 1 : 0);
  };

  Instruction* inst = nullptr;
  if (pool == 0) {
    inst = builder->CreatePoolingMax(ext->GetName(), ext->GetOperand(0));
    set_pooling_attributes(DynCast<PoolingMaxInst>(inst));
  } else {
    inst = builder->CreatePoolingAvg(ext->GetName(), ext->GetOperand(0));
    set_pooling_attributes(DynCast<PoolingAvgInst>(inst));
  }
  return {*inst};
}

static std::vector<Def> ConvertPriorBox(const CAFFEExtensionInst* ext,
                                        IRBuilder* builder) {
  const auto& ty0 = ext->GetOperand(0).GetType();
  const auto& ty1 = ext->GetOperand(1).GetType();
  if (!ty0.IsValid() || !ty1.IsValid()) {
    return {};
  }
  const int h = ty0.GetNumOfElementsInDim(2);
  const int w = ty0.GetNumOfElementsInDim(3);
  const int ih = ty1.GetNumOfElementsInDim(2);
  const int iw = ty1.GetNumOfElementsInDim(3);
  auto aspect_ratios =
      FindAttributeValue(*ext, "aspect_ratio", std::vector<float>{1});
  const auto& min_sizes =
      FindAttributeValue(*ext, "min_size", std::vector<float>{});
  HLCHECK(!min_sizes.empty());
  const auto& max_sizes =
      FindAttributeValue(*ext, "max_size", std::vector<float>{});
  HLCHECK(max_sizes.empty() || max_sizes.size() == min_sizes.size());
  bool flip = FindAttributeValue<bool>(*ext, "flip");
  bool clip = FindAttributeValue<bool>(*ext, "clip");
  std::vector<float> variance =
      FindAttributeValue<std::vector<float>>(*ext, "variance");
  float step = FindAttributeValue<float>(*ext, "step");
  float offset = FindAttributeValue<float>(*ext, "offset");

  {
    std::vector<float> ars;
    ars.reserve(aspect_ratios.size() * 2);
    ars.push_back(1.0F);
    constexpr float error = 1e-6;
    for (auto r : aspect_ratios) {
      bool exists = false;
      for (auto x : ars) {
        exists = fabsf(x - r) < error;
        if (exists) {
          break;
        }
      }
      if (!exists) {
        ars.push_back(r);
        if (flip) {
          ars.push_back(1.0F / r);
        }
      }
    }
    ars.swap(aspect_ratios);
  }

  auto priors_per_point =
      aspect_ratios.size() * min_sizes.size() + max_sizes.size();

  std::vector<int64_t> shape{
      1, 2, 4 * h * w * static_cast<int64_t>(priors_per_point)};
  halo::Type ty{DataType::FLOAT32, shape};
  std::vector<float> data;
  data.reserve(ty.GetTotalNumOfElements());
  const float image_h = static_cast<float>(ih);
  const float image_w = static_cast<float>(iw);
  float step_h = step == 0 ? image_h / static_cast<float>(h) : step;
  float step_w = step == 0 ? image_w / static_cast<float>(w) : step;
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      float center_x = (static_cast<float>(x) + offset) * step_w;
      float center_y = (static_cast<float>(y) + offset) * step_h;
      for (int s = 0, se = min_sizes.size(); s < se; ++s) {
        float min_size = min_sizes[s];
        for (float ar : aspect_ratios) {
          float ar_s = sqrtf(ar);
          float box_width = min_size * ar_s;
          float box_height = min_size / ar_s;
          data.push_back((center_x - box_width / 2) / image_w);  // xmin.
          data.push_back((center_y - box_height / 2) / image_h); // ymin.
          data.push_back((center_x + box_width / 2) / image_w);  // xmax
          data.push_back((center_y + box_height / 2) / image_h); // ymax.

          if (ar == 1.0F && !max_sizes.empty()) {
            float max_size = max_sizes[s];
            float box_width = sqrtf(min_size * max_size);
            float box_height = box_width;
            data.push_back((center_x - box_width / 2) / image_w);  // xmin.
            data.push_back((center_y - box_height / 2) / image_h); // ymin.
            data.push_back((center_x + box_width / 2) / image_w);  // xmax
            data.push_back((center_y + box_height / 2) / image_h); // ymax.
          }
        }
      }
    }
  }
  if (clip) {
    for (auto& x : data) {
      x = std::min(1.F, std::max(x, 0.F));
    }
  }
  // Fill variance.
  HLCHECK(static_cast<int64_t>(data.size() * 2) == ty.GetTotalNumOfElements());
  HLCHECK(variance.size() == 1 || variance.size() == 4);
  for (int i = 0, e = data.size(), v = variance.size(); i < e; ++i) {
    data.push_back(variance[i % v]);
  }
  HLCHECK(static_cast<int64_t>(data.size()) == ty.GetTotalNumOfElements());
  ConstantBuilder cb(ext->GetParent()->GetParent());
  auto c = cb.CreateConstant(ext->GetName(), ty, data.data());
  return {*c};
}

static std::vector<Def> ConvertInnerProduct(const CAFFEExtensionInst* ext,
                                            IRBuilder* builder) {
  HLCHECK(ext->GetNumOfOperands() >= 2 && ext->GetNumOfOperands() <= 3);
  bool has_bias = FindAttributeValue(*ext, "bias_term", true);
  auto input = ext->GetOperand(0);
  const auto& input_type = input.GetType();

  auto op1 = ext->GetOperand(1);
  auto& op1_type = op1.GetType();
  if (!input_type.IsValid() || !op1_type.IsValid()) {
    return {};
  }
  if (IsA<Constant>(op1) && op1_type.GetNumOfDims() > 2) {
    // check if leading dims are ones.
    for (unsigned i = 0, e = op1_type.GetNumOfDims() - 2; i < e; ++i) {
      HLCHECK(op1_type.GetNumOfElementsInDim(i) == 1);
    }
    auto dims = op1_type.GetDimSizes();
    auto dim_a = dims[dims.size() - 2];
    auto dim_b = dims.back();
    op1.GetOwner()->GetResultsTypes()[0] =
        Type{op1_type.GetDataType(), {dim_a, dim_b}};
  }
  HLCHECK(op1_type.GetNumOfDims() == 2);
  bool transpose = FindAttributeValue(*ext, "transpose", false);
  size_t axis = FindAttributeValue(*ext, "axis", 1);
  HLCHECK(axis < input_type.GetNumOfDims());

  ConstantBuilder cb(ext->GetParent()->GetParent());
  HLCHECK(input_type.GetNumOfDims() >= 2);
  if (input_type.GetNumOfDims() > 2) {
    auto new_shape = input_type.GetDimSizes();
    for (size_t i = 1; i < axis; ++i) {
      new_shape[0] *= new_shape[i];
    }
    new_shape[1] = new_shape[axis];
    for (size_t i = axis + 1, e = new_shape.size(); i < e; ++i) {
      new_shape[1] *= new_shape[i];
    }
    new_shape.resize(2);
    Constant* c = cb.CreateConstant(
        ext->GetName() + "_shape",
        Type{DataType::INT64, {static_cast<int64_t>(new_shape.size())}, true},
        new_shape.data());
    input = *builder->CreateReshapeDynamic(ext->GetName() + "_reshape",
                                           {input, *c});
  }
  auto matmul = builder->CreateMatMul(ext->GetName(), {input, op1});
  matmul->SetTransposeB(!transpose); // Caffe uses col-major by default.
  if (has_bias) {
    HLCHECK(ext->GetNumOfOperands() == 3);
    auto op2 = ext->GetOperand(2);
    const auto& op2_type = op2.GetType();
    if (auto rank = op2_type.GetNumOfDims(); IsA<Constant>(op2) && rank > 1) {
      for (unsigned i = 0; i < rank - 1; ++i) {
        HLCHECK(op2_type.GetNumOfElementsInDim(i) == 1);
      }
      op2.GetOwner()->GetResultsTypes()[0] =
          Type{op2_type.GetDataType(),
               {1, op2_type.GetNumOfElementsInDim(rank - 1)}};
    }

    return {*builder->CreateAdd(ext->GetName() + "_bias", {*matmul, op2})};
  }
  return {*matmul};
}

static std::vector<Def> ConvertScale(const CAFFEExtensionInst* ext,
                                     IRBuilder* builder) {
  if (ext->GetNumOfOperands() == 1) {
    return {ext->GetOperand(0)};
  }
  HLCHECK(ext->GetNumOfOperands() > 1);
  auto input = ext->GetOperand(0);
  const auto& input_type = input.GetType();
  if (!input_type.IsValid()) {
    return {};
  }
  bool has_bias = FindAttributeValue(*ext, "bias_term", false);

  auto op1 = ext->GetOperand(1);
  if (!op1.GetType().IsValid()) {
    return {};
  }
  if (has_bias) {
    HLCHECK(ext->GetNumOfOperands() == 3);
    if (!ext->GetOperand(2).GetType().IsValid()) {
      return {};
    }
  } else {
    HLCHECK(ext->GetNumOfOperands() == 2);
  }
  // Check if input is BatchNorm without scaling. If so, fuse with it.
  bool all_consts =
      has_bias && IsA<Constant>(op1) && IsA<Constant>(ext->GetOperand(2));
  if (all_consts && IsA<Instruction>(input) &&
      DynCast<Instruction>(input)->GetOpCode() == OpCode::BATCHNORM) {
    BatchNormInst* bn = DynCast<BatchNormInst>(input);
    bool bn_without_scale = bn->GetNumOfOperands() == 3;
    if (!bn_without_scale && bn->GetNumOfOperands() == 4) {
      auto last_op = bn->GetOperand(3);
      if (IsA<Constant>(last_op)) {
        bn_without_scale = DynCast<Constant>(last_op)->IsScalarOne();
      }
    }
    int ch = input_type.GetNumOfElementsInDim(1);
    auto op2 = ext->GetOperand(2);
    if (bn_without_scale && op1.GetType().GetTotalNumOfElements() == ch &&
        op2.GetType().GetTotalNumOfElements() == ch) {
      std::vector<Def> new_ops{bn->GetOperand(0), op1, op2, bn->GetOperand(1),
                               bn->GetOperand(2)};
      Instruction* new_bn = builder->Clone(*bn, new_ops);
      return {*new_bn};
    }
  }

  // check if operand needs broadcasting.
  int axis = FindAttributeValue(*ext, "axis", 1);
  int ranks = input_type.GetNumOfDims();
  axis = axis < 0 ? ranks + axis : axis;
  auto reshape_if_needed = [builder, axis, &input_type,
                            ext](const Def& def) -> Def {
    const auto& ty = def.GetType();
    std::vector<int64_t> new_shape = ty.GetDimSizes();
    int trailing_ones =
        static_cast<int>(input_type.GetNumOfDims() - ty.GetNumOfDims()) - axis;
    HLCHECK(trailing_ones >= 0);
    if (trailing_ones > 0) {
      for (int i = 0; i < trailing_ones; ++i) {
        new_shape.push_back(1);
      }
      ConstantBuilder cb(ext->GetParent()->GetParent());
      Constant* c = cb.CreateConstant(
          ext->GetName() + "_shape",
          Type{DataType::INT64, {static_cast<int64_t>(new_shape.size())}, true},
          new_shape.data());

      return *builder->CreateReshapeDynamic(
          def.GetDef()->GetName() + "_reshape", {def, *c});
    }
    return def;
  };
  op1 = reshape_if_needed(op1);
  input = *(builder->CreateMul(ext->GetName() + "_scale", {input, op1}));
  if (has_bias) {
    auto op2 = ext->GetOperand(2);
    op2 = reshape_if_needed(op2);
    input = *(builder->CreateAdd(ext->GetName() + "_bias", {input, op2}));
  } else {
    HLCHECK(!has_bias);
  }
  return {input};
}

static std::vector<Def> ConvertPower(const CAFFEExtensionInst* ext,
                                     IRBuilder* builder) {
  HLCHECK(ext->GetNumOfOperands() == 1);
  // computes y = (shift + scale * x) ^ power.

  auto input = ext->GetOperand(0);
  float power = FindAttributeValue(*ext, "power", 1.0F);
  float scale = FindAttributeValue(*ext, "scale", 1.0F);
  float shift = FindAttributeValue(*ext, "shift", .0F);

  ConstantBuilder cb(ext->GetParent()->GetParent());
  if (scale != 1) {
    Constant* c = cb.CreateConstant(ext->GetName() + "_scale",
                                    Type{DataType::FLOAT32, {1}}, &scale);
    input = *(builder->CreateMul(ext->GetName() + "_mul", {input, *c}));
  }
  if (shift != 0) {
    Constant* c = cb.CreateConstant(ext->GetName() + "_shift",
                                    Type{DataType::FLOAT32, {1}}, &shift);
    input = *(builder->CreateAdd(ext->GetName() + "_add", {input, *c}));
  }
  if (power != 1) {
    Constant* c = cb.CreateConstant(ext->GetName() + "_power",
                                    Type{DataType::FLOAT32, {1}}, &shift);
    input = *(builder->CreatePow(ext->GetName() + "_power", {input, *c}));
  }
  input.GetDef()->SetName(ext->GetName());
  return {input};
}

static std::vector<Def> ConvertReshape(const CAFFEExtensionInst* ext,
                                       IRBuilder* builder) {
  HLCHECK(ext->GetNumOfOperands() >= 1);
  auto bottom = ext->GetOperand(0);
  const Type& input_type = bottom.GetType();
  if (!input_type.IsValid()) {
    return {};
  }

  int axis = FindAttributeValue(*ext, "axis", 0);
  int num_axes = FindAttributeValue(*ext, "num_axes", -1);

  if (HasAttribute(*ext, "shape")) {
    const auto& shape = FindAttributeValue(*ext, "shape", std::vector<int>{});
    // TODO(unknown) : handle general cases to reshape in the range of [axis,
    // axis + num_axes]
    HLCHECK(num_axes == -1);
    HLCHECK(axis == 0);
    ConstantBuilder cb(ext->GetParent()->GetParent());
    Constant* c = cb.CreateConstant(
        ext->GetName() + "_shape",
        Type{DataType::INT32, {static_cast<int64_t>(shape.size())}, true},
        shape.data());
    auto new_inst = builder->CreateReshapeDynamic(ext->GetName(), {bottom, *c});
    return {*new_inst};
  }

  return {};
}

static std::vector<Def> ConvertDeConvolution(const CAFFEExtensionInst* ext,
                                             IRBuilder* builder) {
  std::vector<int> pad(2, 0);
  pad = FindAttributeValue(*ext, "pad", pad);
  if (pad.size() == 1) {
    pad.push_back(pad.front());
  }
  pad[0] = FindAttributeValue(*ext, "pad_h", pad[0]);
  pad[1] = FindAttributeValue(*ext, "pad_w", pad[1]);

  std::vector<int> stride(4, 1);
  stride = FindAttributeValue(*ext, "stride", stride);
  if (stride.size() == 1) {
    stride.push_back(stride.front());
  }
  while (stride.size() < 4) {
    stride.insert(stride.begin(), 1);
  }
  stride[2] = FindAttributeValue(*ext, "stride_h", stride[2]);
  stride[3] = FindAttributeValue(*ext, "stride_w", stride[3]);
  std::vector<int> dilation(4, 1);
  const auto& value =
      FindAttributeValue(*ext, "dilation", std::vector<int>{1, 1});
  dilation[2] = value.empty() ? 1 : value[0];
  dilation[3] = (value.size() > 1) ? value[1] : dilation[2];

#if 0
  std::vector<int> kernel_size(4, 1);
  if (const auto& it = attr_map.find("kernel_h"); it != attr_map.end()) {
    kernel_size[2] = ext->GetAttributes()[it->second].get()->GetValueAsInteger();
  } else if (const auto& it = attr_map.find("kernel_size"); it != attr_map.end()) {
    kernel_size[2] = ext->GetAttributes()[it->second].get()->GetValueAsIntegerList()[0];
  }
  if (const auto& it = attr_map.find("kernel_w"); it != attr_map.end()) {
    kernel_size[3] = ext->GetAttributes()[it->second].get()->GetValueAsInteger();
  } else if (const auto& it = attr_map.find("kernel_size"); it != attr_map.end()) {
    const auto& value =
        ext->GetAttributes()[it->second].get()->GetValueAsIntegerList();
    kernel_size[3] = value.size() > 1 ? value[1] : value[0];
  }

  int num_output = 0;
  if (const auto& it = attr_map.find("num_output"); it != attr_map.end()) {
    num_output = ext->GetAttributes()[it->second].get()->GetValueAsInteger();
  }
#endif
  // TODO (unknown) ignore group param in deconv
  int group = FindAttributeValue(*ext, "group", 1);
  bool bias_term = FindAttributeValue(*ext, "bias_term", true);

  HLCHECK(ext->GetNumOfOperands() == 2 || ext->GetNumOfOperands() == 3);
  auto input = ext->GetOperand(0);
  auto weight = ext->GetOperand(1);

  IRObject* new_inst =
      builder->CreateConv2DTranspose(ext->GetName() + "_conv", input, weight);
  Conv2DTransposeInst* deconv_inst = Downcast<Conv2DTransposeInst>(new_inst);
  deconv_inst->SetPaddingLeft(pad[1]);
  deconv_inst->SetPaddingRight(pad[1]);
  deconv_inst->SetPaddingTop(pad[0]);
  deconv_inst->SetPaddingBottom(pad[0]);
  deconv_inst->SetStrides(stride);
  deconv_inst->SetDilations(dilation);
  deconv_inst->SetGroup(group);
  // deconv_inst->SetNumOutput(num_output);
  deconv_inst->SetPadding(Padding::EXPLICIT);
  deconv_inst->SetDataFormat(DataFormat::NCHW);
  deconv_inst->SetFilterFormat(DataFormat::CNHW);

  if (bias_term) {
    auto bias = ext->GetOperand(2);
    const auto& orig_bias = DynCast<Constant>(bias.GetOwner());
    // do broadcast, need to be offline for performance
    auto shape = bias.GetType().GetDimSizes();
    const static int axis = 1; // broadcast on C
    const static int dims = 4; // conv2D
    HLCHECK(shape.size() == 1);
    for (int i = 0; i < axis; ++i) {
      shape.insert(shape.begin(), 1);
    }
    for (int i = 2; i < dims; ++i) {
      shape.emplace_back(1);
    }
    ConstantBuilder cb(ext->GetParent()->GetParent());
    Constant* c = cb.CreateConstant(ext->GetName() + "_bias",
                                    Type{DataType::FLOAT32, shape},
                                    orig_bias->GetRawDataPtr());
    new_inst = builder->CreateAdd(ext->GetName() + "_add", *new_inst, *c);
  }
  return {*new_inst};
}

static std::vector<Def> ConvertTile(const CAFFEExtensionInst* ext,
                                    IRBuilder* builder) {
  HLCHECK(ext->GetNumOfOperands() == 1);
  auto input = ext->GetOperand(0);
  const auto& input_type = input.GetType();

  if (!input_type.IsValid()) {
    return {};
  }
  int dims = input_type.GetNumOfDims();
  int axis = FindAttributeValue(*ext, "axis", 1);
  axis = (axis < 0) ? dims + axis : axis;
  int copies = FindAttributeValue(*ext, "tiles", 1);
  std::vector<int64_t> multipliers(dims, 1);
  HLCHECK(axis < dims);
  multipliers[axis] = copies;
  ConstantBuilder cb(ext->GetParent()->GetParent());

  Constant* c =
      cb.CreateConstant(ext->GetName() + "_tile", Type{DataType::INT64, {dims}},
                        multipliers.data());
  TileInst* new_inst = builder->CreateTile(ext->GetName(), input, *c);
  return {*new_inst};
}

static std::vector<Def> ConvertRelu(const CAFFEExtensionInst* ext,
                                    IRBuilder* builder) {
  HLCHECK(ext->GetNumOfOperands() == 1);
  auto input = ext->GetOperand(0);

  builder->SetInsertAfter(ext);

  float alpha = ext->GetNumOfAttributes() != 0
                    ? ext->GetAttributes()[0]->GetValueAsFloat()
                    : 0;
  if (alpha != 0) {
    auto leakly_relu = builder->CreateLeakyRelu(ext->GetName(), input);
    leakly_relu->SetAlpha(alpha);
    return {*leakly_relu};
  }
  auto relu = builder->CreateRelu(ext->GetName(), input);
  return {*relu};
}

static std::vector<Def> ConvertBatchNorm(const CAFFEExtensionInst* ext,
                                         IRBuilder* builder) {
  HLCHECK(ext->GetNumOfOperands() > 1);
  auto input = ext->GetOperand(0);
  bool has_use_global_stats =
      FindAttributeValue(*ext, "use_global_stats", false);

  if (has_use_global_stats) {
    // use the stored mean/variance estimates.
    auto bn = builder->CreateBatchNorm(ext->GetName(), ext->GetOperands());
    bn->SetDataFormat(DataFormat::NCHW);
    return {*bn};
  }
  // compute mean and var
  auto bn = builder->CreateInstanceNorm(ext->GetName(), ext->GetOperands());
  bn->SetDataFormat(DataFormat::NCHW);
  return {*bn};
}

static std::vector<Def> ConvertCAFFEExtension(
    const CAFFEExtensionInst* caffe_inst, IRBuilder* builder) {
  builder->SetInsertAfter(caffe_inst);
  switch (caffe_inst->GetExtOpCode()) {
    case CAFFEExtOpCode::CONVOLUTION: {
      return ConvertConvolution(caffe_inst, builder);
    }
    case CAFFEExtOpCode::DECONVOLUTION: {
      return ConvertDeConvolution(caffe_inst, builder);
    }
    case CAFFEExtOpCode::DETECTIONOUTPUT: {
      return ConvertDetectionOutput(caffe_inst, builder);
    }
    case CAFFEExtOpCode::ELTWISE: {
      return ConvertEltwise(caffe_inst, builder);
    }
    case CAFFEExtOpCode::INNERPRODUCT: {
      return ConvertInnerProduct(caffe_inst, builder);
    }
    case CAFFEExtOpCode::POOLING: {
      return ConvertPool(caffe_inst, builder);
    }
    case CAFFEExtOpCode::RESHAPE: {
      return ConvertReshape(caffe_inst, builder);
    }
    case CAFFEExtOpCode::SCALE: {
      return ConvertScale(caffe_inst, builder);
    }
    case CAFFEExtOpCode::UPSAMPLE: {
      return ConvertUpsample(caffe_inst, builder);
    }
    case CAFFEExtOpCode::POWER: {
      return ConvertPower(caffe_inst, builder);
    }
    case CAFFEExtOpCode::DROPOUT: {
      return {caffe_inst->GetOperand(0)};
    }
    case CAFFEExtOpCode::FLATTEN: {
      return ConvertFlatten(caffe_inst, builder);
    }
    case CAFFEExtOpCode::NORMALIZE: {
      return ConvertNormalize(caffe_inst, builder);
    }
    case CAFFEExtOpCode::PRIORBOX: {
      return ConvertPriorBox(caffe_inst, builder);
    }
    case CAFFEExtOpCode::TILE: {
      return ConvertTile(caffe_inst, builder);
    }
    case CAFFEExtOpCode::RELU: {
      return ConvertRelu(caffe_inst, builder);
    }
    case CAFFEExtOpCode::BATCHNORM: {
      return ConvertBatchNorm(caffe_inst, builder);
    }
    default: {
      HLCHECK(0 && "Unhandled");
    }
  }
  return std::vector<Def>{};
}

bool CAFFEExtensionLegalizer::RunOnBasicBlock(BasicBlock* bb) {
  IRBuilder builder(bb);
  bool changed = false;
  changed |= AppendReturnInst(bb);
  for (auto& inst_t : *bb) {
    Instruction* inst = inst_t.get();
    if (inst->GetOpCode() == OpCode::EXTENSION) {
      ExtensionInst* ext_inst = Downcast<ExtensionInst>(inst);
      if (ext_inst->GetExtensionKind() ==
          ExtensionInst::ExtensionKind::kExtension_CAFFE) {
        CAFFEExtensionInst* caffe_inst = Downcast<CAFFEExtensionInst>(ext_inst);
        auto new_defs = ConvertCAFFEExtension(caffe_inst, &builder);
        if (!new_defs.empty()) {
          caffe_inst->ReplaceAllUsesWith(new_defs);
        }
      }
    }
  }
  return changed;
}
} // end namespace halo
