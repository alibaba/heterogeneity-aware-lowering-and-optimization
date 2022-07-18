//===- onnxextension_legalizer.cc -----------------------------------------===//
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

#include "halo/lib/transforms/onnxextension_legalizer.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <set>
#include <unordered_set>

#include "halo/api/halo_data.h"
#include "halo/lib/framework/common.h"
#include "halo/lib/ir/common_instructions.h"
#include "halo/lib/ir/constant.h"
#include "halo/lib/ir/extension_instructions.h"
#include "halo/lib/ir/ir_builder.h"
#include "halo/lib/ir/math_instructions.h"
#include "halo/lib/transforms/transforms_util.h"
#include "onnx_parser.h"

namespace halo {

static std::vector<Def> ConvertUnsqueeze(const ONNXExtensionInst* ext,
                                         IRBuilder* builder) {
  auto num_ops = ext->GetNumOfOperands();
  HLCHECK(num_ops == 1 || num_ops == 2);
  auto input = ext->GetOperand(0);
  const Type& input_type = input.GetType();

  if (!input_type.IsValid()) {
    return {};
  }
  std::set<int> axes;
  if (num_ops == 2) {
    const Constant* axes_c = DynCast<Constant>(ext->GetOperand(1));
    if (axes_c == nullptr) {
      return {};
    }
    auto n = axes_c->GetResultType().GetTotalNumOfElements();
    for (int i = 0; i < n; ++i) {
      axes.insert(axes_c->GetDataAsInt64(i));
    }
  } else {
    HLCHECK(ext->GetNumOfAttributes() == 1);
    const Attribute* attr = ext->GetAttributes()[0].get();
    HLCHECK(attr->GetName() == "axes");
    const auto& axes_attr = attr->GetValueAsIntegerList();
    axes.insert(axes_attr.begin(), axes_attr.end());
  }

  int output_rank = static_cast<int>(input_type.GetNumOfDims() + axes.size());

  for (auto d : axes) {
    HLCHECK(d >= -output_rank && d <= output_rank - 1);
  }

  std::vector<int64_t> new_dims(output_rank);
  for (int i = 0, j = 0; i < output_rank; ++i) {
    new_dims[i] = (axes.count(i) != 0 || axes.count(i - output_rank) != 0)
                      ? 1
                      : input_type.GetNumOfElementsInDim(j++);
  }

  ConstantBuilder cb(ext->GetParent()->GetParent());
  Constant* c = cb.CreateConstant(
      ext->GetName() + "_unsqueeze",
      Type{DataType::INT64, {static_cast<int64_t>(new_dims.size())}},
      new_dims.data());
  auto new_inst = builder->CreateReshape(ext->GetName(), {input, *c});
  return {*new_inst};
}

static std::vector<Def> ConvertNonZero(const ONNXExtensionInst* ext,
                                       IRBuilder* builder) {
  auto input = ext->GetOperand(0);
  if (!IsA<Constant>(input)) {
    return {};
  }
  const auto& in_type = input.GetType();
  int64_t rank = in_type.GetNumOfDims();
  std::vector<int64_t> non_zero_indices; // [rank, n]
  const auto& c = DynCast<Constant>(input);
  int64_t n = in_type.GetTotalNumOfElements();
  switch (in_type.GetDataType()) {
    case DataType::INT64: {
      for (int64_t i = 0; i < n; ++i) {
        if (c->GetData<int64_t>(i) != 0) {
          non_zero_indices.push_back(i);
        }
      }
      break;
    }
    case DataType::INT32: {
      for (int64_t i = 0; i < n; ++i) {
        if (c->GetData<int32_t>(i) != 0) {
          non_zero_indices.push_back(i);
        }
      }
      break;
    }
    case DataType::FLOAT32: {
      for (int64_t i = 0; i < n; ++i) {
        if (c->GetData<float>(i) != 0) {
          non_zero_indices.push_back(i);
        }
      }
      break;
    }
    default:
      HLCHECK(0);
      return {input};
  }
  int64_t len = non_zero_indices.size();
  std::vector<int64_t> data(rank * len);
  auto extends = in_type.GetDimSizes();
  int64_t product = 1;
  for (auto it = extends.rbegin(), e = extends.rend(); it != e; ++it) {
    auto t = *it;
    *it = product;
    product *= t;
  }

  for (int64_t i = 0; i < len; ++i) {
    auto idx = non_zero_indices[i];
    // break idx into indices
    for (int j = 0; j < rank; ++j) {
      data[j * len + i] = idx / extends[j];
      idx -= extends[j] * data[j * len + i];
    }
  }
  ConstantBuilder cb(ext->GetParent()->GetParent());
  Type dst_ty{DataType::INT64, std::vector<int64_t>{rank, len}};
  auto new_inst = cb.CreateConstant(ext->GetName(), dst_ty, data.data());
  return {*new_inst};
}

static std::vector<Def> ConvertConstantOfShape(const ONNXExtensionInst* ext,
                                               IRBuilder* builder) {
  auto shape = DynCast<Constant>(ext->GetOperand(0));
  if (shape == nullptr) {
    return {};
  }
  auto dt = ext->GetAttributes()[0]->GetValueAsEnumDataType();
  HLCHECK(shape->GetResultType().GetNumOfDims() == 1);
  auto rank = shape->GetResultType().GetTotalNumOfElements();
  std::vector<int64_t> dims(rank);
  for (int i = 0; i < rank; ++i) {
    dims[i] = shape->GetDataAsInt64(i);
  }
  Type dst_ty{dt, dims};
  ConstantBuilder cb(ext->GetParent()->GetParent());
  const auto& val = ext->GetAttributes()[1];
  HLCHECK(val->GetName() == "value");
  DefaultDataLayout data_layout;
  size_t elem_size = data_layout.Bytes(dt);
  size_t elems_count = dst_ty.GetTotalNumOfElements();
  std::vector<uint8_t> buf(elem_size * elems_count);
  for (size_t i = 0; i < elems_count; ++i) {
    memcpy(&buf[i * elem_size], val->GetDataImpl(), elem_size);
  }
  auto c = cb.CreateConstant(ext->GetName(), dst_ty, buf.data());
  return {*c};
}

static std::vector<Def> ConvertDepthToSpace(const ONNXExtensionInst* ext,
                                            IRBuilder* builder) {
  const auto& input = ext->GetOperand(0);
  const auto& type = input.GetType();
  if (!type.IsValid()) {
    return {};
  }
  HLCHECK(type.GetNumOfDims() == 4);
  auto batch = type.GetNumOfElementsInDim(0);
  auto ch = type.GetNumOfElementsInDim(1);
  auto h = type.GetNumOfElementsInDim(2);
  auto w = type.GetNumOfElementsInDim(3);

  int block_size = FindAttributeValue(*ext, "blocksize", 0);
  HLCHECK(block_size > 0 && ch % (block_size * block_size) == 0);
  auto new_ch = ch / block_size / block_size;
  std::string mode = FindAttributeValue(*ext, "mode", std::string{"DCR"});
  std::transform(mode.begin(), mode.end(), mode.begin(),
                 [](unsigned char c) { return std::toupper(c); });
  HLCHECK(mode == "CRD" || mode == "DCR");
  constexpr int tmp_rank = 6;
  std::vector<int64_t> shape_0;
  std::vector<int32_t> perm;
  std::vector<int64_t> shape_1{batch, new_ch, h * block_size, w * block_size};

  if (mode == "DCR") {
    shape_0 = {batch, block_size, block_size, new_ch, h, w};
    perm = {0, 3, 4, 1, 5, 2}; // NOLINT.
  } else {
    // CDR mode.
    shape_0 = {batch, new_ch, block_size, block_size, h, w};
    perm = {0, 1, 4, 2, 5, 3}; // NOLINT.
  }
  ConstantBuilder c_builder(ext->GetParent()->GetParent());
  const auto& name = input.GetDef()->GetName();
  auto c_shape_0 = c_builder.CreateConstant(
      name + "_shape_0", Type{DataType::INT64, {tmp_rank}}, shape_0);
  auto v = builder->CreateReshape(name + "_reshape_0", input, *c_shape_0);
  auto tr = builder->CreateTranspose(name + "_tr", {*v});
  tr->SetPermutation(perm);
  auto c_shape_1 = c_builder.CreateConstant(
      name + "_shape_1", Type{DataType::INT64, {4}}, shape_1);
  return {*builder->CreateReshape(name + "_reshape_1", *tr, *c_shape_1)};
}

static std::vector<Def> ConvertDynamicQuantize(const ONNXExtensionInst* ext,
                                               IRBuilder* builder) {
  const auto& input = ext->GetOperand(0);
  const auto& name = ext->GetName();
  ConstantBuilder c_builder(ext->GetParent()->GetParent());
  ReduceMinInst* min = builder->CreateReduceMin(name + "_min", {input});
  min->SetKeepDims(false);
  Def min_op = *min;
  Def zero = *c_builder.CreateConstant(
      name + "_zero", halo::Type{DataType::FLOAT32, std::vector<int64_t>{}},
      std::vector<float>{0});
  min_op = *builder->CreateMinimum(name + "_range_min", min_op, zero);
  ReduceMaxInst* max = builder->CreateReduceMax(name + "_max", {input});
  max->SetKeepDims(false);
  Def max_op = *max;
  max_op = *builder->CreateMaximum(name + "_range_max", max_op, zero);
  SubInst* range = builder->CreateSub(name + "_range", max_op, min_op);
  constexpr float q_range = 255;
  auto q_range_c = c_builder.CreateConstant(
      name + "_qrange", halo::Type{DataType::FLOAT32, std::vector<int64_t>{}},
      std::vector<float>{q_range});
  DivInst* scale = builder->CreateDiv(name + "_scale", *range, *q_range_c);
  auto n_zp = builder->CreateDiv(name + "_zp_neg", min_op, *scale);
  auto zp_float = builder->CreateNeg(name + "_zp_float", *n_zp);
  auto zp_round = builder->CreateRound(name + "_zp_round", *zp_float);
  auto zp = builder->CreateFPtoSI(name + "_zp", *zp_round);
  zp->SetDataType(DataType::UINT8);
  auto q = builder->CreateQuantize(name, input, *scale, *zp);
  q->SetSignBit(false);
  q->SetAxis(std::numeric_limits<int32_t>::max());
  return {*q, *scale, *zp};
}

static std::vector<Def> ConvertEyeLike(const ONNXExtensionInst* ext,
                                       IRBuilder* builder) {
  auto type = ext->GetOperand(0).GetType();
  if (!type.IsValid()) {
    return {};
  }
  HLCHECK(type.GetNumOfDims() == 2);
  int k = FindAttributeValue<int>(*ext, "k", 0);
  auto elem_type = ONNXParser::ProcessDataType(
      FindAttributeValue<int>(*ext, "dtype", -1), true /* allow_invalid */);
  if (elem_type == DataType::INVALID) {
    elem_type = type.GetDataType();
  }
  if (elem_type == DataType::INVALID) {
    elem_type = DataType::FLOAT32;
  }

  auto rows = type.GetNumOfElementsInDim(0);
  auto cols = type.GetNumOfElementsInDim(1);
  type = halo::Type{elem_type, {rows, cols}};
  DefaultDataLayout dl;
  auto elem_size = dl.Bytes(elem_type);
  static const float f32 = 1.0F;
  static const int32_t i32 = 1;
  static const double f64 = 1.0;
  static const int64_t i64 = 1;
  static const int8_t i8 = 1;
  static const std::unordered_map<DataType, const void*> bufs{
      {DataType::FLOAT32, &f32},
      {DataType::FLOAT64, &f64},
      {DataType::INT32, &i32},
      {DataType::INT64, &i64},
      {DataType::INT8, &i8}};
  auto it = bufs.find(elem_type);
  HLCHECK(it != bufs.end());
  std::vector<char> data(rows * cols * elem_size);
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      if (c == r + k) {
        memcpy(&data[elem_size * (r * cols + c)], it->second, elem_size);
      }
    }
  }
  ConstantBuilder cb(ext->GetParent()->GetParent());
  auto c = cb.CreateConstant(ext->GetName(), type, data.data());
  return {*c};
}

static std::vector<Def> ConvertPad(const ONNXExtensionInst* ext,
                                   IRBuilder* builder) {
  std::vector<int32_t> paddings;
  PadMode mode = PadMode::CONSTANT;
  float value = 0;
  HLCHECK(ext->GetNumOfAttributes() == 3 || ext->GetNumOfAttributes() == 2);
  for (size_t i = 0; i < ext->GetNumOfAttributes(); ++i) {
    const Attribute* attr = ext->GetAttributes()[i].get();
    if (attr->GetName() == "pads") {
      paddings = attr->GetValueAsIntegerList();
    } else if (attr->GetName() == "mode") {
      mode = attr->GetValueAsEnumPadMode();
    } else {
      if (mode == PadMode::CONSTANT) {
        value = attr->GetValueAsFloat();
      }
    }
  }

  bool all_zeros = true;
  for_each(paddings.begin(), paddings.end(),
           [&all_zeros](int x) { all_zeros &= (x == 0); });
  if (all_zeros) {
    return {ext->GetOperand(0)};
  }

  HLCHECK(value == 0 && mode == PadMode::CONSTANT);
  int64_t rank = paddings.size() / 2;
  std::vector<int32_t> padding_data(paddings.size());
  for (int axis = 0; axis < rank; ++axis) {
    HLCHECK(paddings[axis] >= 0 && paddings[rank + axis] >= 0);
    padding_data[axis * 2] = paddings[axis];
    padding_data[axis * 2 + 1] = paddings[rank + axis];
  }
  ConstantBuilder cb(ext->GetParent()->GetParent());
  auto* pad_amt =
      cb.CreateConstant(ext->GetName() + "_amt",
                        Type{DataType::INT32, {rank, 2}}, padding_data.data());
  auto pad_inst =
      builder->CreatePad(ext->GetName(), {ext->GetOperand(0), *pad_amt});
  return {*pad_inst};
}

static std::vector<Def> ConvertSize(const ONNXExtensionInst* ext,
                                    IRBuilder* builder) {
  const auto& type = ext->GetOperand(0).GetType();
  if (!type.IsValid()) {
    return {};
  }
  auto n = type.GetTotalNumOfElements();
  ConstantBuilder cb(ext->GetParent()->GetParent());
  return {*(cb.CreateConstant(ext->GetName(), Type{DataType::INT64, {1}}, &n))};
}

static std::vector<Def> ConvertSum(const ONNXExtensionInst* ext,
                                   IRBuilder* builder) {
  // Conver to a chain of adds.
  auto n = ext->GetNumOfOperands();
  HLCHECK(n >= 1);
  if (n == 1) {
    return {ext->GetOperand(0)};
  }
  auto op0 = builder->CreateAdd(ext->GetName(), ext->GetOperand(0),
                                ext->GetOperand(1));
  for (unsigned i = 2; i < n; ++i) {
    op0 = builder->CreateAdd(ext->GetName() + std::to_string(i - 1), *op0,
                             ext->GetOperand(i));
  }
  return {*op0};
}

static std::vector<Def> ConvertMaximum(const ONNXExtensionInst* ext,
                                       IRBuilder* builder) {
  // Conver to a chain of maximum.
  auto n = ext->GetNumOfOperands();
  HLCHECK(n >= 1);
  if (n == 1) {
    return {ext->GetOperand(0)};
  }
  auto op0 = builder->CreateMaximum(ext->GetName(), ext->GetOperand(0),
                                    ext->GetOperand(1));
  for (unsigned i = 2; i < n; ++i) {
    op0 = builder->CreateMaximum(ext->GetName() + std::to_string(i - 1), *op0,
                                 ext->GetOperand(i));
  }
  return {*op0};
}

static std::vector<Def> ConvertMinimum(const ONNXExtensionInst* ext,
                                       IRBuilder* builder) {
  // Conver to a chain of minimum.
  auto n = ext->GetNumOfOperands();
  HLCHECK(n >= 1);
  if (n == 1) {
    return {ext->GetOperand(0)};
  }
  auto op0 = builder->CreateMinimum(ext->GetName(), ext->GetOperand(0),
                                    ext->GetOperand(1));
  for (unsigned i = 2; i < n; ++i) {
    op0 = builder->CreateMinimum(ext->GetName() + std::to_string(i - 1), *op0,
                                 ext->GetOperand(i));
  }
  return {*op0};
}

static std::vector<Def> ConvertFlatten(const ONNXExtensionInst* ext,
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
  Constant* c = cb.CreateConstant(ext->GetName() + "_flatten_dims",
                                  Type{DataType::INT32, {2}}, new_dims.data());
  auto new_inst = builder->CreateReshape(ext->GetName(), {input, *c});
  return {*new_inst};
}

static std::vector<Def> ConvertShape(const ONNXExtensionInst* ext,
                                     IRBuilder* builder) {
  auto input = ext->GetOperand(0);
  const Type& input_type = input.GetType();
  if (!input_type.IsValid()) {
    return {};
  }
  std::vector<int64_t> shape;
  for (int64_t i : input_type.GetDimSizes()) {
    shape.push_back(i);
  }
  ConstantBuilder cb(ext->GetParent()->GetParent());
  Constant* c = cb.CreateConstant(
      ext->GetName() + "_shape",
      Type{DataType::INT64, {static_cast<int64_t>(input_type.GetNumOfDims())}},
      shape.data());
  return {*c};
}

static Interpolation ParseInterpolation(const std::string& str) {
  if (str == "nearest") {
    return Interpolation::NEAREST;
  }

  if (str == "linear") {
    return Interpolation::LINEAR;
  }

  if (str == "cubic") {
    return Interpolation::CUBIC;
  }

  return Interpolation::INVALID;
}

static ResizeMode ParseResizeMode(const std::string& str) {
  if (str == "half_pixel") {
    return ResizeMode::HALF_PIXEL;
  }

  if (str == "align_corners") {
    return ResizeMode::ALIGN_CORNERS;
  }

  if (str == "asymmetric") {
    return ResizeMode::ASYMMETRIC;
  }

  return ResizeMode::INVALID;
}

static std::vector<Def> ConvertResize(const ONNXExtensionInst* ext,
                                      IRBuilder* builder) {
  Def input = ext->GetOperand(0);
  if (!input.GetType().IsValid()) {
    return {};
  }

  bool explicit_shape = true;
  auto args = ext->GetNumOfOperands();
  // Scale / Size are the last operand.
  Def scale_size = ext->GetOperand(args - 1);
  if (args >= 3 && IsA<Constant>(scale_size) &&
      scale_size.GetType().GetTotalNumOfElements() == 0) {
    scale_size = ext->GetOperand(args - 2);
  }

  if (Type::IsFloatingPointType(scale_size.GetType().GetDataType())) {
    explicit_shape = false;
  }

  std::vector<Def> ir_operands{input, scale_size};

  std::string co_trs_mode("half_pixel");
  co_trs_mode =
      FindAttributeValue(*ext, "coordinate_transformation_mode", co_trs_mode);

  std::string mode("nearest");
  mode = FindAttributeValue(*ext, "mode", mode);

  builder->SetInsertAfter(ext);
  ResizeInst* resize = builder->CreateResize(ext->GetName(), ir_operands);

  resize->SetInterpolationMode(ParseInterpolation(mode));
  resize->SetMode(ParseResizeMode(co_trs_mode));
  resize->SetAxesMask(-1);
  resize->SetExplicitShape(explicit_shape);

  return {*resize};
}

static std::vector<Def> ConvertSqueeze(const ONNXExtensionInst* ext,
                                       IRBuilder* builder) {
  return ConvertSqueezeImpl<ONNXExtensionInst>(ext, builder, "axes");
}

static std::vector<Def> ConvertCast(const ONNXExtensionInst* ext,
                                    IRBuilder* builder) {
  HLCHECK(ext->GetNumOfAttributes() == 1);
  const Attribute* attr = ext->GetAttributes()[0].get();
  HLCHECK(attr->GetName() == "to");
  // onnx::DataType is not equal to halo::DataType
  const auto& dst_type = ONNXParser::ProcessDataType(attr->GetValueAsInteger());

  auto op0 = ext->GetOperand(0);
  const Type& input_type = op0.GetType();
  if (!input_type.IsValid()) {
    return {};
  }
  const DataType& src_type = input_type.GetDataType();
  if (src_type == dst_type) {
    return {op0};
  }
  if (src_type == DataType::STRING) {
    auto cast = builder->CreateConvertFromString(ext->GetName(), op0);
    cast->SetDataType(dst_type);
    return {*cast};
  }
  if (dst_type == DataType::STRING) {
    auto cast = builder->CreateConvertToString(ext->GetName(), op0);
    cast->SetDataType(dst_type);
    return {*cast};
  }

  if (Type::IsIntegerType(src_type)) {
    if (Type::IsIntegerType(dst_type)) {
      ZExtInst* new_inst = builder->CreateZExt(ext->GetName(), op0);
      new_inst->SetDataType(dst_type);
      return {*new_inst};
    }
    if (Type::IsFloatingPointType(dst_type)) {
      SItoFPInst* new_inst = builder->CreateSItoFP(ext->GetName(), op0);
      new_inst->SetDataType(dst_type);
      return {*new_inst};
    }
  } else if (Type::IsFloatingPointType(src_type)) {
    if (Type::IsIntegerType(dst_type)) {
      FPtoSIInst* new_inst = builder->CreateFPtoSI(ext->GetName(), op0);
      new_inst->SetDataType(dst_type);
      return {*new_inst};
    }
    if (Type::IsFloatingPointType(dst_type)) {
      auto new_inst = builder->CreateFPtoFP(ext->GetName(), op0);
      new_inst->SetDataType(dst_type);
      return {*new_inst};
    }
  } else {
    HLCHECK(0 && "unhandled cast");
  }
  return {};
}

static std::vector<Def> ConvertClip(const ONNXExtensionInst* ext,
                                    IRBuilder* builder) {
  auto num_ops = ext->GetNumOfOperands();
  HLCHECK(num_ops > 0 && num_ops <= 3);
  auto input = ext->GetOperand(0);
  auto in_min = num_ops >= 2 ? ext->GetOperand(1) : Def::GetUndefined();
  auto in_max = num_ops >= 3 ? ext->GetOperand(2) : Def::GetUndefined();
  const auto& undef = Def::GetUndefined();
  ConstantBuilder cb(ext->GetParent()->GetParent());

  if (in_min == undef) {
    float attr_min =
        FindAttributeValue(*ext, "min", std::numeric_limits<float>::lowest());
    in_min = *cb.CreateConstant(ext->GetName() + "_min",
                                Type{DataType::FLOAT32, {1}}, &attr_min);
  }
  if (in_max == undef) {
    float attr_max =
        FindAttributeValue(*ext, "max", std::numeric_limits<float>::max());
    in_max = *cb.CreateConstant(ext->GetName() + "_max",
                                Type{DataType::FLOAT32, {1}}, &attr_max);
  }
  float min = NAN;
  float max = NAN;
  if (const Constant* c = DynCast<Constant>(in_min); c != nullptr) {
    min = c->GetDataAsFloat32(0);
  }
  if (const Constant* c = DynCast<Constant>(in_max); c != nullptr) {
    max = c->GetDataAsFloat32(0);
  }

  constexpr int relu6_max = 6;
  if (min == 0 && max == relu6_max) {
    // TODO(unknown): move relu 6 pattern matching to inst_simplify.
    return {*builder->CreateRelu6(ext->GetName(), ext->GetOperand(0))};
  }
  // If a value is lowest/highest, no need to clip.
  if (min != std::numeric_limits<float>::lowest()) {
    input = *builder->CreateMaximum(
        (num_ops == 2) ? ext->GetName() : ext->GetName() + "_min", input,
        in_min);
  }
  if (max != std::numeric_limits<float>::max()) {
    input = *builder->CreateMinimum(ext->GetName(), input, in_max);
  }
  return {input};
}

static std::vector<Def> ConvertSlice(const ONNXExtensionInst* ext,
                                     IRBuilder* builder) {
  // Operands: input [begin] [end] [axes] [step]
  // For opset 1, begin/end/axes are attributs.
  auto op_num = ext->GetNumOfOperands();
  HLCHECK(op_num >= 1 && op_num != 2 && op_num <= 5);
  ConstantBuilder cb(ext->GetParent()->GetParent());

  // Normalize Slice-1 to Slice.
  if (op_num == 1) {
    HLCHECK(ext->GetNumOfAttributes() >= 2);
    std::vector<int> empty;
    const auto& starts = FindAttributeValue(*ext, "starts", empty);
    const auto& ends = FindAttributeValue(*ext, "ends", empty);
    const auto& axes = FindAttributeValue(*ext, "axes", empty);
    HLCHECK(!starts.empty() && starts.size() == ends.size());
    auto ops = ext->GetOperands();
    int s = starts.size();
    Constant* c_starts = cb.CreateConstant(
        ext->GetName() + "_starts", Type{DataType::INT32, {s}}, starts.data());
    Constant* c_ends = cb.CreateConstant(
        ext->GetName() + "_ends", Type{DataType::INT32, {s}}, ends.data());

    ops.push_back(*c_starts);
    ops.push_back(*c_ends);
    if (!axes.empty()) {
      int s = axes.size();
      Constant* c = cb.CreateConstant(ext->GetName() + "_axes",
                                      Type{DataType::INT32, {s}}, axes.data());
      ops.push_back(*c);
    }
    auto new_inst = builder->Clone(*ext, ops);
    return {*new_inst};
  }
  auto op0 = ext->GetOperand(0);
  auto op_starts = ext->GetOperand(1); // starts
  auto op_ends = ext->GetOperand(2);   // ends

  auto& input_type = op0.GetType();
  if (!input_type.IsValid()) {
    return {};
  }

  if (!IsA<Constant>(op_starts) || !IsA<Constant>(op_ends)) {
    auto op_len =
        builder->CreateSub(ext->GetName() + "_len", op_ends, op_starts);
    std::vector<Def> ops{op0, op_starts, *op_len};

    // Operand [step]: if no step, set step=1
    if (ext->GetNumOfOperands() > 4) {
      ops.push_back(ext->GetOperand(4));
    } else {
      int start_size = op_starts.GetType().GetTotalNumOfElements();
      std::vector<int> steps(input_type.GetNumOfDims(), 1);
      Constant* c_steps =
          cb.CreateConstant(ext->GetName() + "_steps",
                            Type{DataType::INT32, {start_size}}, steps.data());
      ops.push_back(*c_steps);
    }

    // Operand [axes]
    if (ext->GetNumOfOperands() >= 4) {
      ops.push_back(ext->GetOperand(3));
    }

    SliceInst* slice = builder->CreateSlice(ext->GetName(), ops);
    return {*slice};
  }

  int input_dims = input_type.GetNumOfDims();

  Constant* c_starts = DynCast<Constant>(op_starts);
  const auto& starts_type = op_starts.GetType();
  Constant* c_ends = DynCast<Constant>(op_ends);
  const auto& ends_type = op_ends.GetType();

  HLCHECK(ends_type.GetNumOfDims() == starts_type.GetNumOfDims());

  std::set<int32_t> axes; // order matters.

  // If no axes operand, assumes all axes are sliced and steps are 1.
  Def op_axes = Def::GetUndefined();
  Def op_steps = Def::GetUndefined();

  int starts_size = starts_type.GetTotalNumOfElements();
  std::vector<int> steps(input_dims, 1);
  if ((op_num == 3) || (op_num == 4)) {
    Constant* c_steps =
        cb.CreateConstant(ext->GetName() + "_steps",
                          Type{DataType::INT32, {starts_size}}, steps.data());
    op_steps = *c_steps;
    if (op_num == 3) {
      std::vector<int> data(input_dims);
      for (int i = 0; i < input_dims; ++i) {
        axes.insert(i);
        data[i] = i;
      }
      Constant* c_axes =
          cb.CreateConstant(ext->GetName() + "_axes",
                            Type{DataType::INT32, {input_dims}}, data.data());
      op_axes = *c_axes;
    }
  }

  if (op_num >= 4) {
    op_axes = ext->GetOperand(3); // axes
    if (!IsA<Constant>(op_axes)) {
      return {};
    }

    if (op_num > 4) {
      op_steps = ext->GetOperand(4); // steps
      if (!IsA<Constant>(op_steps)) {
        return {};
      }
    }

    Constant* c_axes = DynCast<Constant>(op_axes);
    HLCHECK(Type::IsIntegerType(op_axes.GetType()));
    auto e_a = op_axes.GetType().GetTotalNumOfElements();
    for (int i = 0, tmp = 0; i != e_a; ++i) {
      tmp = c_axes->GetDataAsInt64(i);
      axes.insert(tmp >= 0 ? tmp : tmp + input_dims);
    }

    HLCHECK(!axes.empty());
  }

  std::vector<int32_t> starts(input_dims);
  HLCHECK(Type::IsIntegerType(c_starts->GetResultType()));
  for (int i = 0, j = 0; i < input_dims; ++i) {
    if (axes.count(i) != 0) {
      auto start = c_starts->GetDataAsInt64(j++);
      if (start < 0) {
        start = input_type.GetNumOfElementsInDim(i) + start;
      } else if (start > input_type.GetNumOfElementsInDim(i)) {
        start = input_type.GetNumOfElementsInDim(i);
      }
      starts[i] = start;
    }
  }

  std::vector<int32_t> ends(input_dims);
  HLCHECK(Type::IsIntegerType(ends_type));
  for (int i = 0, j = 0; i < input_dims; ++i) {
    ends[i] = input_type.GetNumOfElementsInDim(i);
    if (axes.count(i) == 0) {
      continue;
    }
    auto end = c_ends->GetDataAsInt64(j++);
    if (end < 0) {
      end += input_type.GetNumOfElementsInDim(i);
    }
    ends[i] = std::min(std::max(end, -1L), static_cast<int64_t>(ends[i]));
  }

  Constant* c_steps = DynCast<Constant>(op_steps);
  HLCHECK(Type::IsIntegerType(op_steps.GetType()));
  for (int i = 0, j = 0; i < input_dims; ++i) {
    if (axes.count(i) != 0) {
      steps[i] = c_steps->GetDataAsInt64(j++);
    }
  }

  // calculate sizes:  1 + (end - start - 1) / step
  std::vector<int> sizes_data;
  std::vector<int> starts_data;
  sizes_data.reserve(axes.size());
  starts_data.reserve(axes.size());
  for (auto axis : axes) {
    starts_data.push_back(starts[axis]);
    sizes_data.push_back((ends[axis] - starts[axis] - 1) / steps[axis] + 1);
    HLCHECK(sizes_data.back() >= 0);
  }
  Constant* c_begins_norm = cb.CreateConstant(
      ext->GetName() + "_starts",
      Type{DataType::INT32, {static_cast<int64_t>(starts_data.size())}},
      starts_data.data());

  Constant* c_sizes_norm = cb.CreateConstant(
      ext->GetName() + "_sizes",
      Type{DataType::INT32, {static_cast<int64_t>(sizes_data.size())}},
      sizes_data.data());

  SliceInst* slice = builder->CreateSlice(
      ext->GetName(), {op0, *c_begins_norm, *c_sizes_norm, op_steps, op_axes});
  return {*slice};
}

static std::vector<Def> ConvertOneHot(const ONNXExtensionInst* ext,
                                      IRBuilder* builder) {
  HLCHECK(ext->GetNumOfAttributes() == 1);
  const Attribute* attr = ext->GetAttributes()[0].get();
  HLCHECK(attr->GetName() == "axis");
  int axis = attr->GetValueAsInteger();
  const std::string& name = ext->GetName();
  ConstantBuilder cb(ext->GetParent()->GetParent());

  auto op0 = ext->GetOperand(0);
  auto op1 = ext->GetOperand(1);
  auto op2 = ext->GetOperand(2);

  if (IsA<Instruction>(op2.GetDef())) {
    const Instruction* op2_inst = Downcast<const Instruction>(op2.GetDef());
    if (op2_inst->GetOpCode() == OpCode::CONCAT) {
      OneHotInst* new_inst = builder->CreateOneHot(
          ext->GetName(), op0, op1, op2, op2_inst->GetOperand(1));

      new_inst->SetAxis(axis);
      return {*new_inst};
    }
  }

  if (IsA<Constant>(op2)) {
    const Constant* values = DynCast<Constant>(op2.GetOwner());

    auto& type = op2.GetType();
    auto data_type = type.GetDataType();
    HLCHECK(type.GetTotalNumOfElements() == 2);

    // split values to on-value and off-value

    const char* ptr = static_cast<const char*>(values->GetRawDataPtr());
    size_t data_type_size = values->GetElementSizeInBytes();
    Type ty{data_type};

    Constant* on_value = cb.CreateConstant(
        name + "_on_value", ty,
        static_cast<const void*>(&ptr[data_type_size])); // NOLINT

    OneHotInst* new_inst = builder->CreateOneHot(ext->GetName(), op0, op1,
                                                 op2 /* off_on */, *on_value);

    new_inst->SetAxis(axis);
    return {*new_inst};
  }

  // Get on value.
  const int32_t one_v = 1;
  auto one = cb.CreateConstant(name + "_one", halo::Type{DataType::INT32, {1}},
                               &one_v);
  auto on_value = builder->CreateSlice("split_on", {op2, *one, *one});
  OneHotInst* new_inst = builder->CreateOneHot(ext->GetName(), op0, op1,
                                               op2 /* off_on */, *on_value);
  new_inst->SetAxis(axis);
  return {*new_inst};
}

static std::vector<Def> ConvertSplit(const ONNXExtensionInst* ext,
                                     IRBuilder* builder) {
  auto op0 = ext->GetOperand(0);
  auto& input_type = op0.GetType();
  if (!input_type.IsValid()) {
    return {};
  }
  int input_dims = input_type.GetNumOfDims();

  HLCHECK(ext->GetNumOfAttributes() == 2);
  const auto& attr = FindAttributeValue(*ext, "axis", 0);
  int axis = (attr < 0) ? attr + input_dims : attr;
  HLCHECK(((axis >= 0) && (axis < input_dims)) && "Invalid axis.");
  std::vector<int> empty;
  const auto& splits = FindAttributeValue(*ext, "split", empty);

  ConstantBuilder cb(ext->GetParent()->GetParent());
  // If no axes operand, assumes all axes are sliced and steps are 1.
  Def op3 = Def::GetUndefined();
  Def op4 = Def::GetUndefined();

  std::vector<int> data(input_dims);
  std::vector<int> steps(input_dims, 1);

  for (int i = 0; i < input_dims; ++i) {
    data[i] = i;
  }
  Constant* c_axes =
      cb.CreateConstant(ext->GetName() + "_axes",
                        Type{DataType::INT32, {input_dims}}, data.data());
  op3 = *c_axes;
  Constant* c_steps =
      cb.CreateConstant(ext->GetName() + "_steps",
                        Type{DataType::INT32, {input_dims}}, steps.data());
  op4 = *c_steps;

  std::vector<Def> ret_v;
  int64_t dim = 0;

  std::vector<int64_t> sizes;
  std::vector<std::vector<int64_t>> sizes_v;
  std::vector<std::vector<int64_t>> starts_v;
  int32_t num_outputs = static_cast<int32_t>(ext->GetNumOfResults());

  if (splits.empty()) {
    for (size_t i = 0; i < input_type.GetNumOfDims(); ++i) {
      dim = input_type.GetNumOfElementsInDim(i);
      if (i == static_cast<size_t>(axis)) {
        dim /= num_outputs;
      }
      sizes.push_back(dim);
    }
    for (int32_t i = 0; i < num_outputs; ++i) {
      sizes_v.push_back(sizes);
    }
  } else {
    for (size_t idx = 0; idx < splits.size(); ++idx) {
      for (size_t i = 0; i < input_type.GetNumOfDims(); ++i) {
        if (i == static_cast<size_t>(axis)) {
          dim = splits[idx];
        } else {
          dim = input_type.GetNumOfElementsInDim(i);
        }
        sizes.push_back(dim);
      }
      sizes_v.push_back(sizes);
      sizes.clear();
    }
  }

  int64_t offset = 0;
  int j = 0;
  for (auto sizes : sizes_v) {
    std::vector<int64_t> starts;
    for (size_t i = 0; i < input_type.GetNumOfDims(); ++i) {
      int64_t value = (i == static_cast<size_t>(axis)) ? offset : 0;
      starts.push_back(value);
    }
    starts_v.push_back(starts);
    offset += sizes[axis];

    Constant* c_begins = cb.CreateConstant(
        ext->GetName() + "_starts_" + std::to_string(j),
        Type{DataType::INT64, {static_cast<int64_t>(input_dims)}},
        starts.data());

    Constant* c_sizes = cb.CreateConstant(
        ext->GetName() + "_sizes_" + std::to_string(j),
        Type{DataType::INT64, {static_cast<int64_t>(input_dims)}},
        sizes.data());

    SliceInst* slice =
        builder->CreateSlice(ext->GetName() + "_slice_" + std::to_string(j),
                             {op0, *c_begins, *c_sizes});

    ret_v.push_back(*slice);
    j++;
  }

  return ret_v;
}

static std::vector<Def> ConvertRange(const ONNXExtensionInst* ext,
                                     IRBuilder* builder) {
  HLCHECK(ext->GetNumOfOperands() == 3);
  const Constant* start = DynCast<Constant>(ext->GetOperand(0));
  const Constant* limit = DynCast<Constant>(ext->GetOperand(1));
  const Constant* delta = DynCast<Constant>(ext->GetOperand(2));

  if (start == nullptr || limit == nullptr || delta == nullptr) {
    return {};
  }
  const auto& elem_type = start->GetResultType().GetDataType();
  int64_t begin = start->GetDataAsInt64(0);
  int64_t end = limit->GetDataAsInt64(0);
  int64_t step = delta->GetDataAsInt64(0);

  ConstantBuilder cb(ext->GetParent()->GetParent());

  auto fill = [&cb, ext](DataType dt, auto data, int64_t start, int64_t limit,
                         int64_t delta) {
    int64_t n =
        std::max(0L, static_cast<int64_t>(std::ceil((limit - start) / delta)));
    data.reserve(n);
    for (int64_t i = start; i != limit; i += delta) {
      data.push_back(i);
    }
    return cb.CreateConstant(ext->GetName(), Type{dt, {n}}, data.data());
  };
  switch (elem_type) {
    case DataType::INT32:
      return {*fill(elem_type, std::vector<int32_t>{}, begin, end, step)};
    case DataType::INT64:
      return {*fill(elem_type, std::vector<int64_t>{}, begin, end, step)};
    case DataType::FLOAT32:
      return {*fill(elem_type, std::vector<float>{}, begin, end, step)};
    default:
      HLCHECK("Unhandled type for range");
  }
  return {};
}

std::vector<Def> ConvertGlobalMaxPooling(const ONNXExtensionInst* ext,
                                         IRBuilder* builder) {
  HLCHECK(ext->GetNumOfOperands() == 1);
  auto input = ext->GetOperand(0);
  if (!input.GetType().IsValid()) {
    return {};
  }
  const auto& input_type = input.GetType();

  HLCHECK(input_type.GetNumOfDims() == 4);
  int kernel_size_h = input_type.GetNumOfElementsInDim(2);
  int kernel_size_w = input_type.GetNumOfElementsInDim(3);

  auto set_pooling_attributes = [&](auto inst) {
    inst->SetKsize({1, 1, kernel_size_h, kernel_size_w});
    inst->SetPaddingLeft(0);
    inst->SetPaddingRight(0);
    inst->SetPaddingTop(0);
    inst->SetPaddingBottom(0);
    inst->SetStrides({1, 1, 1, 1});
    inst->SetPadding(Padding::EXPLICIT);
    inst->SetDataFormat(DataFormat::NCHW);
    inst->SetRoundMode(0);
  };

  Instruction* inst = nullptr;
  inst = builder->CreatePoolingMax(ext->GetName(), ext->GetOperand(0));
  set_pooling_attributes(DynCast<PoolingMaxInst>(inst));
  return {*inst};
}

static std::vector<Def> ConvertHgEngine(const ONNXExtensionInst* ext,
                                        IRBuilder* builder) {
  auto n = ext->GetNumOfOperands();
  HLCHECK(n >= 1);
  int attr_idx = 0;
  // Convert serializedEngine to constant input
  ConstantBuilder cb(ext->GetParent()->GetParent());
  auto engine = ext->GetAttributes()[attr_idx++]->GetValueAsString();
  Type type{DataType::INT8, {static_cast<int64_t>(engine.size())}};
  Constant* serialized_engine = cb.CreateConstant(
      ext->GetName() + "_serialized_engine", type,
      reinterpret_cast<const int8_t*>(engine.c_str())); // NOLINT.

  auto ops = ext->GetOperands();
  ops.push_back(*serialized_engine);
  auto hg_engine = builder->CreateHgEngine(ext->GetName(), ops);

  hg_engine->SetInDataFormat(
      ext->GetAttributes()[attr_idx++]->GetValueAsString());
  hg_engine->SetOutDataFormat(
      ext->GetAttributes()[attr_idx++]->GetValueAsString());
  /// Hgai should define the stringlist to support output_shapes,
  /// Now support one output
  std::string shape_str = ext->GetAttributes()[attr_idx++]->GetValueAsString();
  std::vector<std::vector<int64_t>> output_shapes;
  output_shapes.resize(1);
  SplitStringToInt64List(shape_str, &(output_shapes[0]), ",");
  hg_engine->SetOutputShapes(output_shapes);
  hg_engine->SetInBindingList(
      {ext->GetAttributes()[attr_idx++]->GetValueAsString()});
  hg_engine->SetOutBindingList(
      {ext->GetAttributes()[attr_idx++]->GetValueAsString()});
  hg_engine->SetInTypeList(
      {ext->GetAttributes()[attr_idx++]->GetValueAsString()});
  hg_engine->SetOutTypeList(
      {ext->GetAttributes()[attr_idx++]->GetValueAsString()});

  return {*hg_engine};
}

static std::vector<Def> ConvertHgQuant(const ONNXExtensionInst* ext,
                                       IRBuilder* builder) {
  auto input = ext->GetOperand(0);
  const auto& input_type = input.GetType();
  if (!input_type.IsValid()) {
    return {};
  }
  /// By now the Hgai onnx interfce set in_scale/in_bias to sting
  std::vector<float> in_scale;
  std::vector<float> in_bias;
  in_scale.reserve(1);
  in_bias.reserve(1);
  in_scale.emplace_back(std::stof(ext->GetAttributes()[0]->GetValueAsString()));
  in_bias.emplace_back(std::stof(ext->GetAttributes()[1]->GetValueAsString()));
  int is_per_channel = ext->GetAttributes()[3]->GetValueAsInteger();

  ConstantBuilder cb(ext->GetParent()->GetParent());
  Constant* c_scale = nullptr;
  Constant* c_bias = nullptr;

  if (is_per_channel != 0) {
    // Now support per layer quantize, due to the hgai onnx string format
    HLCHECK(0);
  } else {
    HLCHECK(in_scale.size() == 1);
    HLCHECK(in_bias.size() == 1);
    std::vector<float> scale_data(1, in_scale[0]);
    std::vector<float> bias_data(1, in_bias[0]);
    Type scalar_type{DataType::FLOAT32, std::vector<int64_t>{1}};
    c_scale = cb.CreateConstant(ext->GetName() + "_const_scale", scalar_type,
                                scale_data);
    c_bias = cb.CreateConstant(ext->GetName() + "_const_bias", scalar_type,
                               bias_data);
  }
  auto hg_quant =
      builder->CreateHgQuant(ext->GetName(), {input, *c_scale, *c_bias});
  return {*hg_quant};
}

static std::vector<Def> ConvertIpuOp(const ONNXExtensionInst* ext,
                                     IRBuilder* builder,
                                     const std::string& op) {
  auto op0 = ext->GetOperand(0);
  const auto& type = op0.GetType();
  if (!type.IsValid()) {
    return {};
  }
  auto new_type = type;
  if (op == "IpuAttentionMask") {
    HLCHECK(ext->GetNumOfOperands() == 2);
    const auto& op1_type = ext->GetOperand(1).GetType();
    if (!op1_type.IsValid()) {
      return {};
    }
    HLCHECK(type.GetNumOfDims() == 2);
    auto batch = type.GetNumOfElementsInDim(0);
    auto seq = type.GetNumOfElementsInDim(1);
    new_type = Type{op1_type.GetDataType(), {batch, 1, seq, seq}};
  }
  auto new_inst =
      builder->CreateCustom(ext->GetName(), ext->GetOperands(), 1, op);
  new_inst->GetResultsTypes()[0] = new_type;
  return {*new_inst};
}

static std::vector<Def> ConvertHgDeQuant(const ONNXExtensionInst* ext,
                                         IRBuilder* builder) {
  // HLCHECK(0 && "Wrong ConvertHgDeQuant");
  auto input = ext->GetOperand(0);

  const auto& input_type = input.GetType();
  if (!input_type.IsValid()) {
    return {};
  }

  int attr_idx = 0;
  std::vector<float> in_scale;
  std::vector<float> in_bias;
  in_scale.reserve(1);
  in_bias.reserve(1);
  in_scale.emplace_back(
      std::stof(ext->GetAttributes()[attr_idx++]->GetValueAsString()));
  in_bias.emplace_back(
      std::stof(ext->GetAttributes()[attr_idx++]->GetValueAsString()));
  int is_per_channel = ext->GetAttributes()[attr_idx++]->GetValueAsInteger();

  builder->SetInsertAfter(ext);
  ConstantBuilder cb(ext->GetParent()->GetParent());
  Constant* c_scale = nullptr;
  Constant* c_bias = nullptr;
  if (is_per_channel != 0) {
    HLCHECK(0);
  } else {
    HLCHECK(in_scale.size() == 1);
    HLCHECK(in_bias.size() == 1);
    std::vector<float> scale_data(1, in_scale[0]);
    std::vector<float> bias_data(1, in_bias[0]);
    Type scalar_type{DataType::FLOAT32, std::vector<int64_t>{1}};
    c_scale = cb.CreateConstant(ext->GetName() + "_const_scale", scalar_type,
                                scale_data);
    c_bias = cb.CreateConstant(ext->GetName() + "_const_bias", scalar_type,
                               bias_data);
  }

  auto hg_dequant =
      builder->CreateHgDequant(ext->GetName(), {input, *c_scale, *c_bias});
  return {*hg_dequant};
}

static std::vector<Def> ConvertTFIDFVec(const ONNXExtensionInst* ext,
                                        IRBuilder* builder) {
  int min_gram = FindAttributeValue(*ext, "min_gram_length", 1);
  int max_gram = FindAttributeValue(*ext, "max_gram_length", 1);
  int max_skip = FindAttributeValue(*ext, "max_skip_count", 0);
  const auto& mode = FindAttributeValue(*ext, "mode", TFIDFMode::INVALID);
  const auto& ngram_counts =
      FindAttributeValue<std::vector<int64_t>>(*ext, "ngram_counts", {});
  const auto& ngram_indexes =
      FindAttributeValue<std::vector<int64_t>>(*ext, "ngram_indexes", {});
  const auto& pool_int =
      FindAttributeValue<std::vector<int64_t>>(*ext, "pool_int64s", {});
  const auto& pool_str =
      FindAttributeValue<std::vector<std::string>>(*ext, "pool_strings", {});

  const auto& weights =
      FindAttributeValue<std::vector<float>>(*ext, "weights", {});
  HLCHECK(pool_int.empty() ^ pool_str.empty());
  auto n = pool_int.empty() ? pool_str.size() : pool_int.size();
  HLCHECK(weights.empty() || weights.size() == n);
  ConstantBuilder c_builder(ext->GetParent()->GetParent());

  const auto& name = ext->GetName();
  auto op_pool = c_builder.CreateConstant(
      name + "_pool", Type{DataType::INT64, {static_cast<int64_t>(n)}},
      pool_int.data());
  auto op_cnts = c_builder.CreateConstant(
      name + "_cnts",
      Type{DataType::INT64, {static_cast<int64_t>(ngram_counts.size())}},
      ngram_counts.data());
  auto op_indices = c_builder.CreateConstant(
      name + "_indices",
      Type{DataType::INT64, {static_cast<int64_t>(ngram_indexes.size())}},
      ngram_indexes.data());
  auto op_weight =
      weights.empty()
          ? Def::GetUndefined()
          : Def{c_builder.CreateConstant(
                    name + "_weight",
                    Type{DataType::FLOAT32, {static_cast<int64_t>(n)}},
                    weights.data()),
                0};

  auto new_inst = builder->CreateTFIDFVectorize(
      ext->GetName(),
      {ext->GetOperand(0), *op_pool, *op_cnts, *op_indices, op_weight});
  new_inst->SetMaxGramLength(max_gram);
  new_inst->SetMinGramLength(min_gram);
  new_inst->SetMaxSkip(max_skip);
  new_inst->SetMode(mode);
  new_inst->SetMaxIdx(
      *std::max_element(ngram_indexes.begin(), ngram_indexes.end()));
  return {*new_inst};
}

static bool FixupTranspose(TransposeInst* inst) {
  if (inst->GetPermutation().empty() &&
      inst->GetOperand(0).GetType().IsValid()) {
    // When permutation attribute is empty, revese all axes.
    int rank = inst->GetOperand(0).GetType().GetNumOfDims();
    std::vector<int> perm(rank);
    for (int i = rank - 1; i >= 0; --i) {
      perm[rank - i - 1] = i;
    }
    inst->SetPermutation(perm);
    return true;
  }
  return false;
}

static bool FixupLoopBody(LoopInst* inst) {
  // For ONNX, the loop has 2 + N inputs: (iter_num, condition, loop vars...),
  // 1 + N + K outputs: (cond, loop vars ...,  scan_outputs...).
  // The first argument (loop_count) also serves as trip iterator.
  // Parser omits the "cond" in return.

  // Avoid re-entry.
  if (HasAttribute(*inst, "halo_fixedup")) {
    return false;
  }
  inst->AddOneAttribute(Attribute::CreateBool("halo_fixedup", true));

  auto body = inst->GetBody();
  HLCHECK(body->Args().size() >= 2);
  Argument* trip_arg = body->arg_begin()->get();
  auto lc_cnt = body->Args().size() - 2;
  auto return_inst = body->GetReturnInst();
  HLCHECK(return_inst->GetNumOfOperands() >= lc_cnt);
  auto scan_output_cnt = return_inst->GetNumOfOperands() - lc_cnt;
  HLCHECK(scan_output_cnt >= 0 && scan_output_cnt <= lc_cnt);

  // Mark scan outputs
  if (!HasAttribute(*return_inst, "halo_scan_output_cnt")) {
    return_inst->AddOneAttribute(
        Attribute::CreateInteger("halo_scan_output_cnt", scan_output_cnt));
  }

  if (trip_arg->GetNumberOfUses() == 0) {
    return false;
  }
  // Add an argument with init value of zero.
  Type ty{DataType::INT32, {}};
  ArgumentBuilder arg_builder(body);
  auto arg_i = arg_builder.CreateArgument(inst->GetName() + "_i", ty);
  HLCHECK(trip_arg->GetResultType().IsScalar());
  ConstantBuilder const_builder(body);
  int one = 1;
  int zero = 0;
  auto c_one = const_builder.CreateConstant(inst->GetName() + "_one", ty, &one);
  auto c_zero = const_builder.CreateConstant(inst->GetName() + "_z", ty, &zero);
  // Create add(init, one).
  IRBuilder builder(body);
  builder.SetInsertBefore(body->begin()->get());
  auto inc = builder.CreateAdd(inst->GetName() + "_inc", *arg_i, *c_one);
  auto attr = Attribute::CreateBool("halo_loop", true);
  inc->AddOneAttribute(
      std::move(attr)); // Mark the instruction as loop carried.
  trip_arg->ReplaceAllUsesWith({*inc});
  inst->AddOneOperand(*c_zero);
  return true;
}

enum LSTMArgIndex {
  LSTM_ARG_X_IDX = 0,
  LSTM_ARG_W_IDX = 1,
  LSTM_ARG_R_IDX = 2,
  LSTM_ARG_B_IDX = 3,
  LSTM_ARG_SEQUENCE_LENGTH_IDX = 4,
  LSTM_ARG_INITIAL_H_IDX = 5,
  LSTM_ARG_INITIAL_C_IDX = 6,
  LSTM_ARG_P_IDX = 7
};

enum LSTMLayout {
  LSTM_LAYOUT_NORMAL = 0,
  LSTM_LAYOUT_TRANSFORMED = 1,
};

Direction DecodeLSTMDirection(const std::string& key) {
  std::string key_lower = key;
  std::transform(key.begin(), key.end(), key_lower.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  static const std::unordered_map<std::string, Direction> enum_map{
      {"forward", Direction::FORWARD},
      {"reverse", Direction::REVERSE},
      {"bidirectional", Direction::BIDIRECTIONAL},
  };

  auto it = enum_map.find(key_lower);
  return it == enum_map.end() ? Direction::INVALID : it->second;
}

enum OptionalArgumentState {
  NORMAL_VALUE_PROVIDED,
  EMPTY_VALUE_PROVIDED,
  NOT_PROVIDED
};

OptionalArgumentState GetStateOfOptionalArgument(const IRObject* obejct,
                                                 size_t idx) {
  if (obejct->GetNumOfOperands() <= idx) {
    return NOT_PROVIDED;
  }
  if (Def::GetUndefined() == obejct->GetOperand(idx)) {
    return EMPTY_VALUE_PROVIDED;
  }
  return NORMAL_VALUE_PROVIDED;
}

static Def ConvertRNNBias(const ONNXExtensionInst& ext, DataType elem_type,
                          int num_directions, int num_gates, int hidden_size,
                          IRBuilder* builder) {
  // B not specified
  auto num_operands = ext.GetNumOfOperands();
  ConstantBuilder c_builder(ext.GetParent()->GetParent());

  int32_t len = num_gates * hidden_size;
  if (num_operands <= LSTM_ARG_B_IDX) {
    Type type(elem_type, {num_directions, len});
    std::string name = ext.GetName() + "_B";
    return *c_builder.SplatConstantZero(name, type);
  }
  builder->SetInsertBefore(&ext);
  auto b = ext.GetOperand(LSTM_ARG_B_IDX);
  halo::Type ty{DataType::INT32, {2}};
  const auto& base_name = b.GetDef()->GetName();
  auto s0 = c_builder.CreateConstant(base_name + "_s0", ty,
                                     std::vector<int32_t>{0, 0});
  auto s1 = c_builder.CreateConstant(base_name + "_s1", ty,
                                     std::vector<int32_t>{0, len});
  auto l = c_builder.CreateConstant(base_name + "_len", ty,
                                    std::vector<int32_t>{num_directions, len});
  auto steps = c_builder.CreateConstant(base_name + "_step", ty,
                                        std::vector<int32_t>{1, 1});

  auto axis = c_builder.CreateConstant(base_name + "_ax", ty,
                                       std::vector<int32_t>{0, 1});
  auto b0 = builder->CreateSlice(base_name + "_w", {b, *s0, *l, *steps, *axis});
  auto b1 = builder->CreateSlice(base_name + "_r", {b, *s1, *l, *steps, *axis});
  return *builder->CreateAdd(base_name + "_wr", {*b0, *b1});
}

static std::vector<Def> ConvertLSTM(const ONNXExtensionInst* ext,
                                    IRBuilder* builder) {
  size_t num_operands = ext->GetNumOfOperands();
  HLCHECK(num_operands > LSTM_ARG_R_IDX && "Missing required arguments");

  const Def& op_x = ext->GetOperand(LSTM_ARG_X_IDX);
  const Type& type_x = op_x.GetType();

  if (!type_x.IsValid()) {
    return {};
  }

  DataType dtype_x = type_x.GetDataType();

  const Def& op_r = ext->GetOperand(2);
  const Type& type_r = op_r.GetType();

  if (!type_r.IsValid()) {
    return {};
  }

  ConstantBuilder c_builder(ext->GetParent()->GetParent());

  int64_t seq_length = type_x.GetNumOfElementsInDim(0);
  int64_t batch_size = type_x.GetNumOfElementsInDim(1);

  int layout = FindAttributeValue<int>(*ext, "layout", LSTM_LAYOUT_NORMAL);
  if (LSTM_LAYOUT_NORMAL != layout) {
    std::swap(seq_length, batch_size);
  }

  int32_t num_directions = type_r.GetNumOfElementsInDim(0);
  int64_t hidden_size = type_r.GetNumOfElementsInDim(2);

  std::vector<Def> operands = ext->GetOperands();

  auto b =
      ConvertRNNBias(*ext, dtype_x, num_directions, 4, hidden_size, builder);

  if (num_operands <= LSTM_ARG_B_IDX) {
    // B not specified
    operands.push_back(b);
  } else {
    operands[LSTM_ARG_B_IDX] = b;
  }

  // sequence_lens not specified
  auto state_sequence_lens =
      GetStateOfOptionalArgument(ext, LSTM_ARG_SEQUENCE_LENGTH_IDX);
  if (NORMAL_VALUE_PROVIDED != state_sequence_lens) {
    std::vector<int32_t> bytes(batch_size, static_cast<int32_t>(seq_length));
    const Type type(DataType::INT32, {batch_size});
    std::string name = ext->GetName() + "_sequence_lens";
    Constant* constant = c_builder.CreateConstant(name, type, bytes.data());

    if (NOT_PROVIDED == state_sequence_lens) {
      operands.push_back(*constant);
    } else {
      operands.at(LSTM_ARG_SEQUENCE_LENGTH_IDX) = *constant;
    }
  }

  auto supply_zeros_default = [&](size_t arg_idx, const char* suffix,
                                  const Type& type) {
    auto state = GetStateOfOptionalArgument(ext, arg_idx);

    if (NORMAL_VALUE_PROVIDED != state) {
      DefaultDataLayout data_layout;
      size_t num_bytes =
          data_layout.Bytes(type.GetDataType(), type.GetTotalNumOfElements());
      std::vector<uint8_t> bytes(num_bytes);
      std::string name = ext->GetName() + suffix;
      Constant* constant = c_builder.CreateConstant(name, type, bytes.data());

      if (NOT_PROVIDED == state) {
        operands.push_back(*constant);
      } else { // EMPTY_VALUE_PROVIDED
        operands.at(arg_idx) = *constant;
      }
    }
  };

  Type type_initial_h(dtype_x, {num_directions, batch_size, hidden_size});
  supply_zeros_default(LSTM_ARG_INITIAL_H_IDX, "_initial_h", type_initial_h);

  Type type_initial_c(dtype_x, {num_directions, batch_size, hidden_size});
  supply_zeros_default(LSTM_ARG_INITIAL_C_IDX, "_initial_c", type_initial_c);

  Type type_p(dtype_x, {num_directions, 3 * hidden_size});
  supply_zeros_default(LSTM_ARG_P_IDX, "_p", type_p);

  builder->SetInsertAfter(ext);

  LSTMInst* lstm = builder->CreateLSTM(ext->GetName(), operands);

  lstm->SetHiddenSize(FindAttributeValue<int>(*ext, "hidden_size", 1));
  lstm->SetLayout(FindAttributeValue<int>(*ext, "layout", 0));

  std::string direction_key("FORWARD");
  direction_key = FindAttributeValue(*ext, "direction", direction_key);

  lstm->SetDirection(DecodeLSTMDirection(direction_key));
  lstm->SetWeightFormat(RNNWeightFormat::LDGOI);
  lstm->SetGateOrder(RNNGateOrder::IOFC);
  return {Def{lstm, 0}, Def{lstm, 1}, Def{lstm, 2}};
}

static std::vector<Def> ConvertRNN(const ONNXExtensionInst* ext,
                                   IRBuilder* builder) {
  size_t num_operands = ext->GetNumOfOperands();
  HLCHECK(num_operands > LSTM_ARG_R_IDX && "Missing required arguments");

  const Def& op_x = ext->GetOperand(LSTM_ARG_X_IDX);
  const Type& type_x = op_x.GetType();

  if (!type_x.IsValid()) {
    return {};
  }

  std::string direction_key("FORWARD");
  direction_key = FindAttributeValue(*ext, "direction", direction_key);
  auto direction = DecodeLSTMDirection(direction_key);
  int num_directions = direction == halo::Direction::BIDIRECTIONAL ? 2 : 1;
  int32_t hidden_size = FindAttributeValue<int>(*ext, "hidden_size", -1);

  DataType dtype_x = type_x.GetDataType();

  ConstantBuilder c_builder(ext->GetParent()->GetParent());

  int64_t seq_length = type_x.GetNumOfElementsInDim(0);
  int64_t batch_size = type_x.GetNumOfElementsInDim(1);

  int layout = FindAttributeValue<int>(*ext, "layout", LSTM_LAYOUT_NORMAL);
  if (LSTM_LAYOUT_NORMAL != layout) {
    std::swap(seq_length, batch_size);
  }

  std::vector<Def> operands(1 + LSTM_ARG_INITIAL_H_IDX + 1,
                            Def::GetUndefined());
  for (unsigned i = 0; i < num_operands; ++i) {
    operands[i] = ext->GetOperand(i);
  }
  builder->SetInsertBefore(ext);
  bool is_gru = ext->GetExtOpCode() == ONNXExtOpCode::GRU;
  int num_gates = is_gru ? 3 : 1;
  operands[LSTM_ARG_B_IDX] = ConvertRNNBias(*ext, dtype_x, num_directions,
                                            num_gates, hidden_size, builder);
  Instruction* rnn = is_gru ? static_cast<Instruction*>(
                                  builder->CreateGRU(ext->GetName(), operands))
                            : builder->CreateRNN(ext->GetName(), operands);
  auto set_attr = [hidden_size, ext, direction](auto inst) {
    inst->SetHiddenSize(hidden_size);
    inst->SetLayout(FindAttributeValue<int>(*ext, "layout", 0));
    inst->SetDirection(direction);
    inst->SetWeightFormat(RNNWeightFormat::LDGOI);
  };
  if (is_gru) {
    auto gru = DynCast<GRUInst>(rnn);
    set_attr(gru);
    gru->SetGateOrder(RNNGateOrder::URO);
  } else {
    set_attr(DynCast<RNNInst>(rnn));
  }
  return {Def{rnn, 0}, Def{rnn, 1}};
}

static std::vector<Def> ConvertSCE(const ONNXExtensionInst* ext,
                                   IRBuilder* builder) {
  auto score =
      builder->CreateLogSoftmax(ext->GetName() + "_1", {ext->GetOperand(0)});
  score->SetAxis(1);
  auto ops = ext->GetOperands();
  ops[0] = *score;
  auto nlll =
      builder->CreateNegativeLogLikelihoodLoss(ext->GetName() + "_0", ops);
  int ignored_idx = FindAttributeValue<int>(*ext, "ignore_index", -1);
  ReductionMode mode =
      FindAttributeValue(*ext, "reduction", ReductionMode::MEAN);
  nlll->SetIgnored(ignored_idx);
  nlll->SetReduction(mode);
  return {*nlll, *score};
}

static std::vector<Def> ConvertONNXExtension(const ONNXExtensionInst* onnx_inst,
                                             IRBuilder* builder) {
  builder->SetInsertAfter(onnx_inst);

  switch (onnx_inst->GetExtOpCode()) {
    case ONNXExtOpCode::MAX: {
      return ConvertMaximum(onnx_inst, builder);
    }
    case ONNXExtOpCode::MIN: {
      return ConvertMinimum(onnx_inst, builder);
    }
    case ONNXExtOpCode::CAST: {
      return ConvertCast(onnx_inst, builder);
    }
    case ONNXExtOpCode::CLIP: {
      return ConvertClip(onnx_inst, builder);
    }
    case ONNXExtOpCode::SHAPE: {
      return ConvertShape(onnx_inst, builder);
    }
    case ONNXExtOpCode::SLICE: {
      return ConvertSlice(onnx_inst, builder);
    }
    case ONNXExtOpCode::RESIZE: {
      return ConvertResize(onnx_inst, builder);
    }
    case ONNXExtOpCode::SQUEEZE: {
      return ConvertSqueeze(onnx_inst, builder);
    }
    case ONNXExtOpCode::UNSQUEEZE: {
      return ConvertUnsqueeze(onnx_inst, builder);
    }
    case ONNXExtOpCode::DEPTHTOSPACE: {
      return ConvertDepthToSpace(onnx_inst, builder);
    }
    case ONNXExtOpCode::DROPOUT: {
      return {onnx_inst->GetOperand(0)};
    }
    case ONNXExtOpCode::DYNAMICQUANTIZELINEAR: {
      return ConvertDynamicQuantize(onnx_inst, builder);
    }
    case ONNXExtOpCode::CONSTANTOFSHAPE: {
      return ConvertConstantOfShape(onnx_inst, builder);
    }
    case ONNXExtOpCode::EYELIKE: {
      return ConvertEyeLike(onnx_inst, builder);
    }
    case ONNXExtOpCode::NONZERO: {
      return ConvertNonZero(onnx_inst, builder);
    }
    case ONNXExtOpCode::FLATTEN: {
      return ConvertFlatten(onnx_inst, builder);
    }
    case ONNXExtOpCode::IPUATTENTIONMASK: {
      return ConvertIpuOp(onnx_inst, builder, "IpuAttentionMask");
    }
    case ONNXExtOpCode::IPUGELU: {
      return ConvertIpuOp(onnx_inst, builder, "IpuGelu");
    }
    case ONNXExtOpCode::IDENTITY: {
      return {onnx_inst->GetOperand(0)};
    }
    case ONNXExtOpCode::ONEHOT: {
      return ConvertOneHot(onnx_inst, builder);
    }
    case ONNXExtOpCode::RANGE: {
      return ConvertRange(onnx_inst, builder);
    }
    case ONNXExtOpCode::SIZE: {
      return ConvertSize(onnx_inst, builder);
    }
    case ONNXExtOpCode::SUM: {
      return ConvertSum(onnx_inst, builder);
    }
    case ONNXExtOpCode::PAD: {
      return ConvertPad(onnx_inst, builder);
    }
    case ONNXExtOpCode::SPLIT: {
      return ConvertSplit(onnx_inst, builder);
    }
    case ONNXExtOpCode::GLOBALMAXPOOL: {
      return ConvertGlobalMaxPooling(onnx_inst, builder);
    }
    case ONNXExtOpCode::HGQUANT: {
      return ConvertHgQuant(onnx_inst, builder);
    }
    case ONNXExtOpCode::HGDEQUANT: {
      return ConvertHgDeQuant(onnx_inst, builder);
    }
    case ONNXExtOpCode::HGENGINE: {
      return ConvertHgEngine(onnx_inst, builder);
    }
    case ONNXExtOpCode::TFIDFVEC: {
      return ConvertTFIDFVec(onnx_inst, builder);
    }
    case ONNXExtOpCode::RNN:
    case ONNXExtOpCode::GRU: {
      return ConvertRNN(onnx_inst, builder);
    }
    case ONNXExtOpCode::LSTM: {
      return ConvertLSTM(onnx_inst, builder);
    }
    case ONNXExtOpCode::SOFTMAXCROSSENTROPY: {
      return ConvertSCE(onnx_inst, builder);
    }
    default: {
      HLCHECK(0 && "Unhandled");
    }
  }
  return std::vector<Def>{};
}

bool ONNXExtensionLegalizer::RunOnBasicBlock(BasicBlock* bb) {
  IRBuilder builder(bb);
  bool changed = false;
  changed |= AppendReturnInst(bb);
  for (auto& inst_t : *bb) {
    Instruction* inst = inst_t.get();
    if (inst->GetOpCode() == OpCode::EXTENSION) {
      ExtensionInst* ext_inst = Downcast<ExtensionInst>(inst);
      if (ext_inst->GetExtensionKind() ==
          ExtensionInst::ExtensionKind::kExtension_ONNX) {
        ONNXExtensionInst* onnx_inst = Downcast<ONNXExtensionInst>(ext_inst);
        auto new_defs = ConvertONNXExtension(onnx_inst, &builder);
        if (!new_defs.empty()) {
          onnx_inst->ReplaceAllUsesWith(new_defs);
        }
      }
    } else if (inst->GetOpCode() == OpCode::LOOP) {
      changed |= FixupLoopBody(DynCast<LoopInst>(inst));
    } else if (inst->GetOpCode() == OpCode::TRANSPOSE) {
      changed |= FixupTranspose(DynCast<TransposeInst>(inst));
    }
  }
  return changed;
}

} // end namespace halo
