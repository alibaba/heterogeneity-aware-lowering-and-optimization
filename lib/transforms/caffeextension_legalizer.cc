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

#include "halo/api/halo_data.h"
#include "halo/lib/framework/common.h"
#include "halo/lib/ir/attribute.h"
#include "halo/lib/ir/common_instructions.h"
#include "halo/lib/ir/extension_instructions.h"
#include "halo/lib/ir/ir_builder.h"
#include "halo/lib/transforms/transforms_util.h"

namespace halo {

static void CreateAttributeMap(const CAFFEExtensionInst* ext,
                               std::unordered_map<std::string, int>* attr_map) {
  int idx = 0;
  for (const auto& it : ext->GetAttributes()) {
    attr_map->emplace(it->GetName(), idx++);
  }
}

static void GetAttributeValue(float* value, const Attribute& attr) {
  *value = attr.GetValueAsFloat();
}

static void GetAttributeValue(bool* value, const Attribute& attr) {
  *value = attr.GetValueAsBool();
}

static void GetAttributeValue(int* value, const Attribute& attr) {
  *value = attr.GetValueAsInteger();
}

template <typename T>
static T FindAttributeValue(const CAFFEExtensionInst* ext,
                            const std::string& name, const T& default_val) {
  T ret_val = default_val;
  for (const auto& it : ext->GetAttributes()) {
    if (it->GetName() == name) {
      GetAttributeValue(&ret_val, *it);
    }
  }
  return ret_val;
}

static std::vector<Def> ConvertConvolution(const CAFFEExtensionInst* ext,
                                           IRBuilder* builder) {
  std::unordered_map<std::string, int> attr_map;
  CreateAttributeMap(ext, &attr_map);

  std::vector<int> pad(2, 0);
  if (const auto& it = attr_map.find("pad"); it != attr_map.end()) {
    const auto& pad_list =
        ext->GetAttributes()[it->second].get()->GetValueAsIntegerList();
    if (!pad_list.empty()) {
      pad[0] = pad_list.front();
      pad[1] = pad_list.back();
    }
  }
  if (const auto& it = attr_map.find("pad_h"); it != attr_map.end()) {
    pad[0] = ext->GetAttributes()[it->second].get()->GetValueAsInteger();
  }
  if (const auto& it = attr_map.find("pad_w"); it != attr_map.end()) {
    pad[1] = ext->GetAttributes()[it->second].get()->GetValueAsInteger();
  }

  std::vector<int> stride(4, 1);
  if (const auto& it = attr_map.find("stride_h"); it != attr_map.end()) {
    stride[2] = ext->GetAttributes()[it->second].get()->GetValueAsInteger();
  } else if (const auto& it = attr_map.find("stride"); it != attr_map.end()) {
    const auto& value =
        ext->GetAttributes()[it->second].get()->GetValueAsIntegerList();
    stride[2] = value.empty() ? 1 : value[0];
  }
  if (const auto& it = attr_map.find("stride_w"); it != attr_map.end()) {
    stride[3] = ext->GetAttributes()[it->second].get()->GetValueAsInteger();
  } else if (const auto& it = attr_map.find("stride"); it != attr_map.end()) {
    const auto& value =
        ext->GetAttributes()[it->second].get()->GetValueAsIntegerList();
    stride[3] = value.empty() ? 1 : value.back();
  }

  std::vector<int> dilation(4, 1);
  if (const auto& it = attr_map.find("dilation"); it != attr_map.end()) {
    const auto& value =
        ext->GetAttributes()[it->second].get()->GetValueAsIntegerList();
    // N and C set to 1
    dilation[2] = value.empty() ? 1 : value[0];
    dilation[3] = (value.size() > 1) ? value[1] : dilation[2];
  }
  int group = 1;
  if (const auto& it = attr_map.find("group"); it != attr_map.end()) {
    group = ext->GetAttributes()[it->second].get()->GetValueAsInteger();
  }

  bool bias_term = true;
  if (const auto& it = attr_map.find("bias_term"); it != attr_map.end()) {
    bias_term = ext->GetAttributes()[it->second].get()->GetValueAsBool();
  }

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
  CreateAttributeMap(ext, &attr_map);
  std::vector<OpCode> ops = {OpCode::MUL, OpCode::ADD, OpCode::MAXIMUM};
  OpCode op_code = OpCode::ADD;
  if (auto it = attr_map.find("operation"); it != attr_map.end()) {
    int value = ext->GetAttributes()[it->second].get()->GetValueAsInteger();
    op_code = ops[value];
  }
  if (auto it = attr_map.find("coeff");
      it != attr_map.end() && op_code == OpCode::ADD) {
    const auto& coeff =
        ext->GetAttributes()[it->second].get()->GetValueAsFloatList();
    if (coeff.size() == 2) {
      // TODO (unknown) support more case
      HLCHECK(coeff[0] == 1.0 && coeff[1] == -1.0);
      op_code = OpCode::SUB;
    }
  }
  auto new_inst = builder->CreateBinary(ext->GetName(), lhs, rhs, op_code);
  return {*new_inst};
}

static std::vector<Def> ConvertPool(const CAFFEExtensionInst* ext,
                                    IRBuilder* builder) {
  HLCHECK(ext->GetNumOfOperands() == 1);
  auto input = ext->GetOperand(0);
  const auto& input_type = input.GetType();
  bool global_pooling = FindAttributeValue<bool>(ext, "global_pooling", false);
  int stride = FindAttributeValue(ext, "stride", 1);
  int kernel_size_h = FindAttributeValue(ext, "kernel_size", -1);
  int kernel_size_w = kernel_size_h;
  int pool = FindAttributeValue(ext, "pool", 0);
  HLCHECK(pool == 0 || pool == 1);

  int pad = FindAttributeValue(ext, "pad", 0);

  if (global_pooling) {
    if (!input_type.IsValid()) {
      return {};
    }
    kernel_size_h = input_type.GetNumOfElementsInDim(2);
    kernel_size_w = input_type.GetNumOfElementsInDim(3);

    stride = 1;
    pad = 0;
  }

  auto set_pooling_attributes = [&](auto inst) {
    inst->SetKsize({1, 1, kernel_size_h, kernel_size_w});
    inst->SetPaddingLeft(pad);
    inst->SetPaddingRight(pad);
    inst->SetPaddingTop(pad);
    inst->SetPaddingBottom(pad);
    inst->SetStrides({1, 1, stride, stride});
    inst->SetPadding(Padding::EXPLICIT);
    inst->SetDataFormat(DataFormat::NCHW);
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

static std::vector<Def> ConvertInnerProduct(const CAFFEExtensionInst* ext,
                                            IRBuilder* builder) {
  HLCHECK(ext->GetNumOfOperands() >= 2 && ext->GetNumOfOperands() <= 3);
  bool has_bias = FindAttributeValue(ext, "bias_term", true);
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
  bool transpose = FindAttributeValue(ext, "transpose", false);
  size_t axis = FindAttributeValue(ext, "axis", 1);
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
        Type{DataType::INT64, {static_cast<int64_t>(new_shape.size())}},
        new_shape.data());
    input = *builder->CreateReshape(ext->GetName() + "_reshape", {input, *c});
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
  HLCHECK(ext->GetNumOfOperands() > 1);
  auto input = ext->GetOperand(0);
  const auto& input_type = input.GetType();
  if (!input_type.IsValid()) {
    return {};
  }
  bool has_bias = FindAttributeValue(ext, "bias_term", false);

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
  int axis = FindAttributeValue(ext, "axis", 1);
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
          Type{DataType::INT64, {static_cast<int64_t>(new_shape.size())}},
          new_shape.data());

      return *builder->CreateReshape(def.GetDef()->GetName() + "_reshape",
                                     {def, *c});
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
  float power = FindAttributeValue(ext, "power", 1.0F);
  float scale = FindAttributeValue(ext, "scale", 1.0F);
  float shift = FindAttributeValue(ext, "shift", .0F);

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
  return {input};
}

static std::vector<Def> ConvertReshape(const CAFFEExtensionInst* ext,
                                       IRBuilder* builder) {
  HLCHECK(ext->GetNumOfOperands() == 1);
  auto bottom = ext->GetOperand(0);
  const Type& input_type = bottom.GetType();
  if (!input_type.IsValid()) {
    return {};
  }

  std::unordered_map<std::string, int> attr_map;
  CreateAttributeMap(ext, &attr_map);
  int axis = 0;
  int num_axes = -1;
  if (auto it = attr_map.find("axis"); it != attr_map.end()) {
    axis = ext->GetAttributes()[it->second].get()->GetValueAsInteger();
  }
  if (auto it = attr_map.find("num_axes"); it != attr_map.end()) {
    num_axes = ext->GetAttributes()[it->second].get()->GetValueAsInteger();
  }

  if (auto it = attr_map.find("shape"); it != attr_map.end()) {
    const auto& shape =
        ext->GetAttributes()[it->second].get()->GetValueAsIntegerList();
    // TODO(unknown) : handle general cases to reshape in the range of [axis,
    // axis + num_axes]
    HLCHECK(num_axes == -1);
    HLCHECK(axis == 0);
    const auto& orig_shape = input_type.GetDimSizes();
    std::vector<int32_t> new_shape;
    int induced_index = -1;
    int64_t new_num_elements = 1;
    auto orig_num_elements = input_type.GetTotalNumOfElements();
    for (int i = 0, e = shape.size(); i != e; ++i) {
      if (shape[i] == 0) {
        new_shape.push_back(static_cast<int32_t>(orig_shape[i]));
      } else if (shape[i] == -1) {
        HLCHECK(induced_index == -1 &&
                "only one dimension is expected to be -1.");
        induced_index = i;
        new_shape.push_back(1);
      } else {
        new_shape.push_back(shape[i]);
      }
      new_num_elements *= new_shape.back();
    }
    if (induced_index != -1) {
      HLCHECK(orig_num_elements % new_num_elements == 0);
      new_shape[induced_index] = orig_num_elements / new_num_elements;
    } else {
      HLCHECK(orig_num_elements == new_num_elements);
    }
    ConstantBuilder cb(ext->GetParent()->GetParent());
    Constant* c = cb.CreateConstant(
        ext->GetName() + "_shape",
        Type{DataType::INT32, {static_cast<int64_t>(new_shape.size())}},
        new_shape.data());
    auto new_inst = builder->CreateReshape(ext->GetName(), {bottom, *c});
    return {*new_inst};
  }
  return {};
}

static std::vector<Def> ConvertDeConvolution(const CAFFEExtensionInst* ext,
                                             IRBuilder* builder) {
  std::unordered_map<std::string, int> attr_map;
  CreateAttributeMap(ext, &attr_map);
  std::vector<int> pad(2, 0);
  if (const auto& it = attr_map.find("pad_h"); it != attr_map.end()) {
    pad[0] = ext->GetAttributes()[it->second].get()->GetValueAsInteger();
  } else if (const auto& it = attr_map.find("pad"); it != attr_map.end()) {
    pad[0] = ext->GetAttributes()[it->second].get()->GetValueAsIntegerList()[0];
  }
  if (const auto& it = attr_map.find("pad_w"); it != attr_map.end()) {
    pad[1] = ext->GetAttributes()[it->second].get()->GetValueAsInteger();
  } else if (const auto& it = attr_map.find("pad"); it != attr_map.end()) {
    const auto& value =
        ext->GetAttributes()[it->second].get()->GetValueAsIntegerList();
    pad[1] = value.size() > 1 ? value[1] : value[0];
  }

  std::vector<int> stride(4, 1);
  if (const auto& it = attr_map.find("stride_h"); it != attr_map.end()) {
    stride[2] = ext->GetAttributes()[it->second].get()->GetValueAsInteger();
  } else if (const auto& it = attr_map.find("stride"); it != attr_map.end()) {
    stride[2] =
        ext->GetAttributes()[it->second].get()->GetValueAsIntegerList()[0];
  }
  if (const auto& it = attr_map.find("stride_w"); it != attr_map.end()) {
    stride[3] = ext->GetAttributes()[it->second].get()->GetValueAsInteger();
  } else if (const auto& it = attr_map.find("stride"); it != attr_map.end()) {
    stride[3] =
        ext->GetAttributes()[it->second].get()->GetValueAsIntegerList()[0];
  }

  std::vector<int> dilation(4, 1);
  if (const auto& it = attr_map.find("dilation"); it != attr_map.end()) {
    const auto& value =
        ext->GetAttributes()[it->second].get()->GetValueAsIntegerList();
    // N and C set to 1
    dilation[2] = value.empty() ? 1 : value[0];
    dilation[3] = (value.size() > 1) ? value[1] : dilation[2];
  }
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
  int group = 1;
  if (const auto& it = attr_map.find("group"); it != attr_map.end()) {
    group = ext->GetAttributes()[it->second].get()->GetValueAsInteger();
  }

  bool bias_term = true;
  if (const auto& it = attr_map.find("bias_term"); it != attr_map.end()) {
    bias_term = ext->GetAttributes()[it->second].get()->GetValueAsBool();
  }

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
  int axis = FindAttributeValue(ext, "axis", 1);
  axis = (axis < 0) ? dims + axis : axis;
  int copies = FindAttributeValue(ext, "tiles", 1);
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
    case CAFFEExtOpCode::TILE: {
      return ConvertTile(caffe_inst, builder);
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
