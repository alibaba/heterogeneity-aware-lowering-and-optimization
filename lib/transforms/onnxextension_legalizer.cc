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
#include <unordered_set>

#include "halo/api/halo_data.h"
#include "halo/lib/framework/common.h"
#include "halo/lib/ir/common_instructions.h"
#include "halo/lib/ir/constant.h"
#include "halo/lib/ir/extension_instructions.h"
#include "halo/lib/ir/ir_builder.h"
#include "halo/lib/transforms/transforms_util.h"
#include "onnx_parser.h"

namespace halo {

static std::vector<Def> ConvertUnsqueeze(const ONNXExtensionInst* ext,
                                         IRBuilder* builder) {
  HLCHECK(ext->GetNumOfOperands() == 1);
  auto input = ext->GetOperand(0);
  const Type& input_type = input.GetType();

  if (!input_type.IsValid()) {
    return {};
  }

  HLCHECK(ext->GetNumOfAttributes() == 1);
  const Attribute* attr = ext->GetAttributes()[0].get();
  HLCHECK(attr->GetName() == "axes");
  std::vector<int> axis = attr->GetValueAsIntegerList();
  std::vector<int64_t> new_dims(input_type.GetDimSizes());
  if (new_dims.empty()) {
    // for scalar type, make its shape as [1].
    HLCHECK(input_type.GetTotalNumOfElements() == 1);
    new_dims.push_back(1);
  } else {
    for (auto& a : axis) {
      if (a < 0) {
        a += input_type.GetNumOfDims();
      }
      new_dims.insert(new_dims.begin() + a, 1);
    }
  }

  ConstantBuilder cb(ext->GetParent()->GetParent());
  Constant* c = cb.CreateConstant(
      ext->GetName() + "_unsqueeze",
      Type{DataType::INT64, {static_cast<int64_t>(new_dims.size())}},
      new_dims.data());
  builder->SetInsertAfter(ext);
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
  auto input = ext->GetOperand(0);
  const Type& input_type = input.GetType();
  if (!input_type.IsValid()) {
    return {};
  }
  if (!IsA<Constant>(input)) {
    return {};
  }

  HLCHECK(input_type.GetDataType() == DataType::INT64);
  auto dt = ext->GetAttributes()[0]->GetValueAsEnumDataType();

  const Constant* shape = DynCast<Constant>(input);
  HLCHECK(shape->GetResultType().GetNumOfDims() == 1);
  auto ranks = shape->GetResultType().GetTotalNumOfElements();
  std::vector<int64_t> dims(ranks);
  for (int i = 0; i < ranks; ++i) {
    dims[i] = shape->GetData<int64_t>(i);
  }
  Type dst_ty{dt, dims};
  ConstantBuilder cb(ext->GetParent()->GetParent());
  Constant* c = nullptr;
  const auto& val = ext->GetAttributes()[1];
  HLCHECK(val->GetName() == "value");
  switch (dt) {
    case DataType::INT32: {
      std::vector<int32_t> data(dst_ty.GetTotalNumOfElements(),
                                val->GetValueAsInteger());
      c = cb.CreateConstant(ext->GetName(), dst_ty, data.data());
      break;
    }
    case DataType::INT64: {
      std::vector<int64_t> data(dst_ty.GetTotalNumOfElements(),
                                val->GetValueAsInteger());
      c = cb.CreateConstant(ext->GetName(), dst_ty, data.data());
      break;
    }
    case DataType::FLOAT32: {
      std::vector<float> data(dst_ty.GetTotalNumOfElements(),
                              val->GetValueAsFloat());
      c = cb.CreateConstant(ext->GetName(), dst_ty, data.data());
      break;
    }
    default: {
    }
  }
  if (c != nullptr) {
    return {*c};
  }
  return {};
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
  builder->SetInsertAfter(ext);
  auto pad_inst =
      builder->CreatePad(ext->GetName(), {ext->GetOperand(0), *pad_amt});
  return {*pad_inst};
}

static std::vector<Def> ConvertSum(const ONNXExtensionInst* ext,
                                   IRBuilder* builder) {
  // Conver to a chain of adds.
  auto n = ext->GetNumOfOperands();
  HLCHECK(n >= 2);
  builder->SetInsertAfter(ext);
  auto op0 = builder->CreateAdd(ext->GetName(), ext->GetOperand(0),
                                ext->GetOperand(1));
  for (unsigned i = 2; i < n; ++i) {
    op0 = builder->CreateAdd(ext->GetName() + std::to_string(i - 1), *op0,
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
  builder->SetInsertAfter(ext);
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

  builder->SetInsertAfter(ext);
  auto op0 = ext->GetOperand(0);
  const Type& input_type = op0.GetType();
  if (!input_type.IsValid()) {
    return {};
  }
  const DataType& src_type = input_type.GetDataType();
  if (src_type == dst_type) {
    return {op0};
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
  }
  HLCHECK(0 && "unhandled cast");
  return {};
}

static std::vector<Def> ConvertSlice(const ONNXExtensionInst* ext,
                                     IRBuilder* builder) {
  // Operands: input [begin] [end] [axes] [step]
  // For opset 1, begin/end/axes are attributs.
  auto op_num = ext->GetNumOfOperands();
  HLCHECK(op_num >= 1 && op_num != 2 && op_num <= 5);
  builder->SetInsertAfter(ext);
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

  builder->SetInsertAfter(ext);

  if (!IsA<Constant>(op_starts) || !IsA<Constant>(op_ends)) {
    auto op_len =
        builder->CreateSub(ext->GetName() + "_len", op_ends, op_starts);
    std::vector<Def> ops{op0, op_starts, *op_len};

    if (ext->GetNumOfOperands() >= 4) {
      ops.push_back(ext->GetOperand(4));
    }
    if (ext->GetNumOfOperands() > 4) {
      ops.push_back(ext->GetOperand(4));
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

  std::unordered_set<int32_t> axes;

  // If no axes operand, assumes all axes are sliced and steps are 1.
  Def op_axes = Def::GetUndefined();
  Def op_steps = Def::GetUndefined();

  std::vector<int> steps(input_dims, 1);
  if ((op_num == 3) || (op_num == 4)) {
    Constant* c_steps =
        cb.CreateConstant(ext->GetName() + "_steps",
                          Type{DataType::INT32, {input_dims}}, steps.data());
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

  // calculate sizes: -((start - end) / step)
  std::vector<int> sizes_data;
  std::vector<int> starts_data;
  sizes_data.reserve(axes.size());
  starts_data.reserve(axes.size());
  for (auto axis : axes) {
    starts_data.push_back(starts[axis]);
    sizes_data.push_back(-((starts[axis] - ends[axis]) / steps[axis]));
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

  builder->SetInsertAfter(ext);
  auto op0 = ext->GetOperand(0);
  auto op1 = ext->GetOperand(1);
  auto op2 = ext->GetOperand(2);

  if (IsA<Instruction>(op2.GetDef())) {
    const Instruction* op2_inst = Downcast<const Instruction>(op2.GetDef());
    if (op2_inst->GetOpCode() == OpCode::CONCAT) {
      OneHotInst* new_inst = builder->CreateOneHot(ext->GetName(), op0, op1,
                                                   op2_inst->GetOperand(0),
                                                   op2_inst->GetOperand(1));

      new_inst->SetAxis(axis);
      return {*new_inst};
    }
  }

  if (IsA<Constant>(op2)) {
    const Constant* values = DynCast<Constant>(op2.GetOwner());

    auto& type = op2.GetType();
    auto data_type = type.GetDataType();

    // split values to on-value and off-value

    const char* ptr = static_cast<const char*>(values->GetRawDataPtr());
    size_t data_type_size =
        values->GetElementSizeInBytes() / type.GetTotalNumOfElements();
    Type ty{data_type};

    Constant* off_value = cb.CreateConstant(name + "_off_value", ty,
                                            static_cast<const void*>(ptr));
    Constant* on_value = cb.CreateConstant(
        name + "_on_value", ty,
        static_cast<const void*>(&ptr[data_type_size])); // NOLINT

    OneHotInst* new_inst =
        builder->CreateOneHot(ext->GetName(), op0, op1, *off_value, *on_value);

    new_inst->SetAxis(axis);
    return {*new_inst};
  }

  HLCHECK(0 && "unhandled OneHot");
  return {};
}

static std::vector<Def> ConvertGatherElements(const ONNXExtensionInst* ext,
                                              IRBuilder* builder) {
  HLCHECK(ext->GetNumOfOperands() == 2);
  HLCHECK(ext->GetNumOfAttributes() == 1);
  const Attribute* attr = ext->GetAttributes()[0].get();
  HLCHECK(attr->GetName() == "axis");

  auto input_op = ext->GetOperand(0);
  auto idx_op = ext->GetOperand(1);
  auto const& input_type = input_op.GetType();
  auto const& idx_type = idx_op.GetType();

  if (!input_type.IsValid() || !idx_type.IsValid()) {
    return {};
  }

  const auto& input_shape = input_type.GetDimSizes();
  auto idx_shape = idx_type.GetDimSizes();

  int axis = attr->GetValueAsInteger();
  axis = axis < 0 ? static_cast<int>(idx_shape.size()) + axis : axis;

  // if idx_shape[i] and input_shape[i] are 1 for all dims except for
  // "axis", it can be converted to Gather. Otherwise, if idx_shape is a
  // result of broadcasting, the input of broadcasting might be converted.
  bool can_be_gather = input_shape.size() == idx_shape.size();
  for (unsigned i = 0, e = input_shape.size(); can_be_gather && i < e; ++i) {
    can_be_gather &= (input_shape[i] == idx_shape[i] && input_shape[i] == 1) ||
                     (i == static_cast<unsigned>(axis));
  }
  if (!can_be_gather) {
    // try to check if input_shape is a result of broadcasting.
    if (IsA<Instruction>(idx_op) &&
        DynCast<Instruction>(idx_op)->GetOpCode() == OpCode::EXPANDDIMS) {
      ExpandDimsInst* exp_dim = DynCast<ExpandDimsInst>(idx_op);
      idx_op = exp_dim->GetOperand(0);
      idx_shape = idx_op.GetType().GetDimSizes(); // FIXME: more checks
      can_be_gather = true;
    }
  }
  if (can_be_gather) {
    ConstantBuilder cb(ext->GetParent()->GetParent());
    std::vector<int64_t> new_dims{idx_shape[axis]};
    Constant* c = cb.CreateConstant(
        ext->GetName() + "_shape",
        Type{DataType::INT64, {static_cast<int64_t>(new_dims.size())}},
        new_dims.data());

    builder->SetInsertAfter(ext);
    auto reshape =
        builder->CreateReshape(ext->GetName() + "_reshape", idx_op, *c);
    auto new_inst = builder->CreateGather(ext->GetName(), {input_op, *reshape});
    new_inst->SetAxis(axis);
    return {*new_inst};
  }
  return {};
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
  const Attribute* attr = ext->GetAttributes()[0].get();
  HLCHECK(attr->GetName() == "axis");
  int axis = attr->GetValueAsInteger();
  axis = (axis < 0) ? axis + input_dims : axis;
  HLCHECK(((axis >= 0) && (axis < input_dims)) && "Invalid axis.");

  const Attribute* attr1 = ext->GetAttributes()[1].get();
  HLCHECK(attr1->GetName() == "split");
  std::vector<int> splits = attr1->GetValueAsIntegerList();

  builder->SetInsertAfter(ext);

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
  std::vector<int64_t> starts;
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
    }
  }

  int64_t offset = 0;
  for (auto sizes : sizes_v) {
    for (size_t i = 0; i < input_type.GetNumOfDims(); ++i) {
      int64_t value = (i == static_cast<size_t>(axis)) ? offset : 0;
      starts.push_back(value);
    }
    starts_v.push_back(starts);
    offset += sizes[axis];

    Constant* c_begins = cb.CreateConstant(
        ext->GetName() + "_starts",
        Type{DataType::INT32, {static_cast<int64_t>(input_dims)}},
        starts.data());

    Constant* c_sizes = cb.CreateConstant(
        ext->GetName() + "_sizes",
        Type{DataType::INT32, {static_cast<int64_t>(input_dims)}},
        sizes.data());

    SliceInst* slice = builder->CreateSlice(
        ext->GetName(), {op0, *c_begins, *c_sizes, op4, op3});

    ret_v.push_back(*slice);
  }

  return ret_v;
}

void SplitString(const std::string& s, std::vector<int64_t>* v,
                 const std::string& c) {
  std::string::size_type pos2 = s.find(c);
  std::string::size_type pos1 = 0;

  while (std::string::npos != pos2) {
    v->push_back(std::stol(s.substr(pos1, pos2 - pos1)));

    pos1 = pos2 + c.size();
    pos2 = s.find(c, pos1);
  }
  if (pos1 != s.length()) {
    v->push_back(std::stol(s.substr(pos1)));
  }
}

static std::vector<Def> ConvertHgEngine(const ONNXExtensionInst* ext,
                                        IRBuilder* builder) {
  auto n = ext->GetNumOfOperands();
  HLCHECK(n >= 1);
  builder->SetInsertAfter(ext);
  int attr_idx = 0;
  auto hg_engine = builder->CreateHgEngine(ext->GetName(), ext->GetOperands());

  hg_engine->SetSerializedEngine(
      ext->GetAttributes()[attr_idx++]->GetValueAsString());
  hg_engine->SetInDataFormat(
      ext->GetAttributes()[attr_idx++]->GetValueAsString());
  hg_engine->SetOutDataFormat(
      ext->GetAttributes()[attr_idx++]->GetValueAsString());
  /// Hgai should define the stringlist to support output_shapes,
  /// Now support one output
  std::string shape_str = ext->GetAttributes()[attr_idx++]->GetValueAsString();
  std::vector<std::vector<int64_t>> output_shapes;
  output_shapes.resize(1);
  SplitString(shape_str, &(output_shapes[0]), ",");
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
  int attr_idx = 0;
  std::vector<float> in_scale;
  std::vector<float> in_bias;
  in_scale.reserve(1);
  in_bias.reserve(1);
  in_scale.emplace_back(
      std::stof(ext->GetAttributes()[attr_idx++]->GetValueAsString()));
  in_bias.emplace_back(
      std::stof(ext->GetAttributes()[attr_idx++]->GetValueAsString()));
  std::string qtype = ext->GetAttributes()[attr_idx++]->GetValueAsString();
  int is_per_channel = ext->GetAttributes()[attr_idx++]->GetValueAsInteger();

  attr_idx += 1;
  std::string in_data_format =
      ext->GetAttributes()[attr_idx++]->GetValueAsString();
  std::string out_data_format =
      ext->GetAttributes()[attr_idx++]->GetValueAsString();

  // get output channel size
  int channel_idx = 3;
  if (in_data_format == "NC" || in_data_format == "NCHW") {
    channel_idx = 1;
  }
  int channel_size = input_type.GetDimSizes()[channel_idx];

  builder->SetInsertAfter(ext);
  ConstantBuilder cb(ext->GetParent()->GetParent());
  Constant* c_scale = nullptr;
  Constant* c_bias = nullptr;

  if (is_per_channel != 0) {
    HLCHECK(in_scale.size() == static_cast<size_t>(channel_size));
    HLCHECK(in_bias.size() == static_cast<size_t>(channel_size));
    Type type{DataType::FLOAT32, std::vector<int64_t>{channel_size}};
    c_scale =
        cb.CreateConstant(ext->GetName() + "_const_scale", type, in_scale);
    c_bias = cb.CreateConstant(ext->GetName() + "_const_bias", type, in_bias);
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

  // convert to mul+bias+round+cast+clip
  // Mul
  input = *(builder->CreateMul(ext->GetName() + "_scale", {input, *c_scale}));
  input = *(builder->CreateAdd(ext->GetName() + "_bias", {input, *c_bias}));
  // round
  input = *(builder->CreateRound(ext->GetName() + "_round", {input}));
  // cast
  DataType dst_type;
  int num_bits = halo::kEightBits;
  if (qtype == "int8") {
    dst_type = DataType::INT8;
  } else if (qtype == "uint8") {
    dst_type = DataType::UINT8;
  } else if (qtype == "int16") {
    dst_type = DataType::INT16;
    num_bits = kSixteenBits;
  } else if (qtype == "uint16") {
    dst_type = DataType::UINT16;
    num_bits = kSixteenBits;
  } else {
    HLCHECK(0 && "Wrong qtype");
  }

  FPtoSIInst* cast_inst =
      builder->CreateFPtoSI(ext->GetName() + "_cast", {input});
  cast_inst->SetDataType(dst_type);

  // clip = Minimum(Maximum(op, hi), lo)
  int hi = 0;
  int lo = 0;
  // get data range.
  if (qtype == "int8" || qtype == "int16") {
    hi = static_cast<int>(std::pow(2, num_bits - 1)) - 1;
    lo = -hi;
  } else {
    hi = static_cast<int>(std::pow(2, num_bits)) - 1;
    lo = 0;
  }

  Type type_int{DataType::INT32, std::vector<int64_t>{1}};
  std::vector<int> hi_data(1, hi);
  std::vector<int> lo_data(1, lo);
  Constant* c_hi = cb.CreateConstant(ext->GetName() + "_hi", type_int, hi_data);
  Constant* c_lo = cb.CreateConstant(ext->GetName() + "_lo", type_int, lo_data);
  // Maximum
  input = *(builder->CreateBinary(ext->GetName() + "_max", *cast_inst, *c_hi,
                                  OpCode::MAXIMUM));
  // Minimumåå
  input = *(builder->CreateBinary(ext->GetName() + "_min", input, *c_lo,
                                  OpCode::MINIMUM));

  if (in_data_format != out_data_format) {
    // transpose
    TransposeInst* new_transpose =
        builder->CreateTranspose(ext->GetName() + "_transpose", {input});
    if ((in_data_format == "NCHW") && (out_data_format == "NHWC")) {
      std::vector<int> nchw2nhwc{0, 2, 3, 1};
      new_transpose->SetPermutation(nchw2nhwc);
    } else if ((in_data_format == "NHWC") && (out_data_format == "NCHW")) {
      std::vector<int> nhwc2nchw{0, 3, 1, 2};
      new_transpose->SetPermutation(nhwc2nchw);
    }
    return {*new_transpose};
  }

  return {input};
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
  std::string in_data_format =
      ext->GetAttributes()[attr_idx++]->GetValueAsString();
  std::string out_data_format =
      ext->GetAttributes()[attr_idx++]->GetValueAsString();

  // get output channel size
  int channel_idx = 3;
  if (in_data_format == "NC" || in_data_format == "NCHW" ||
      (input_type.GetDimSizes().size() <= 2)) {
    channel_idx = 1;
  }
  int channel_size = input_type.GetDimSizes()[channel_idx];
  builder->SetInsertAfter(ext);
  ConstantBuilder cb(ext->GetParent()->GetParent());
  Constant* c_scale = nullptr;
  Constant* c_bias = nullptr;
  Type type{DataType::FLOAT32, std::vector<int64_t>{channel_size}};
  if (is_per_channel != 0) {
    HLCHECK(in_scale.size() == static_cast<size_t>(channel_size));
    HLCHECK(in_bias.size() == static_cast<size_t>(channel_size));
    c_scale =
        cb.CreateConstant(ext->GetName() + "_const_scale", type, in_scale);
    c_bias = cb.CreateConstant(ext->GetName() + "_const_bias", type, in_bias);
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

  // convert to cast+mul+bias

  // cast
  SItoFPInst* cast_inst =
      builder->CreateSItoFP(ext->GetName() + "_cast", {input});
  cast_inst->SetDataType(DataType::FLOAT32);

  // Mul
  input =
      *(builder->CreateMul(ext->GetName() + "_scale", {*cast_inst, *c_scale}));
  // bias_add
  input = *(builder->CreateAdd(ext->GetName() + "_bias", {input, *c_bias}));

  if (in_data_format != out_data_format) {
    // transpose
    TransposeInst* new_transpose =
        builder->CreateTranspose(ext->GetName() + "_transpose", {input});
    if ((in_data_format == "NCHW") && (out_data_format == "NHWC")) {
      std::vector<int> nchw2nhwc{0, 2, 3, 1};
      new_transpose->SetPermutation(nchw2nhwc);
    } else if ((in_data_format == "NHWC") && (out_data_format == "NCHW")) {
      std::vector<int> nhwc2nchw{0, 3, 1, 2};
      new_transpose->SetPermutation(nhwc2nchw);
    }
    return {*new_transpose};
  }

  return {input};
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

static std::vector<Def> ConvertONNXExtension(const ONNXExtensionInst* onnx_inst,
                                             IRBuilder* builder) {
  switch (onnx_inst->GetExtOpCode()) {
    case ONNXExtOpCode::CAST: {
      return ConvertCast(onnx_inst, builder);
    }
    case ONNXExtOpCode::SHAPE: {
      return ConvertShape(onnx_inst, builder);
    }
    case ONNXExtOpCode::SLICE: {
      return ConvertSlice(onnx_inst, builder);
    }
    case ONNXExtOpCode::SQUEEZE: {
      return ConvertSqueeze(onnx_inst, builder);
    }
    case ONNXExtOpCode::GATHERELEMENTS: {
      return ConvertGatherElements(onnx_inst, builder);
    }
    case ONNXExtOpCode::UNSQUEEZE: {
      return ConvertUnsqueeze(onnx_inst, builder);
    }
    case ONNXExtOpCode::DROPOUT: {
      return {onnx_inst->GetOperand(0)};
    }
    case ONNXExtOpCode::CONSTANTOFSHAPE: {
      return ConvertConstantOfShape(onnx_inst, builder);
    }
    case ONNXExtOpCode::NONZERO: {
      return ConvertNonZero(onnx_inst, builder);
    }
    case ONNXExtOpCode::FLATTEN: {
      return ConvertFlatten(onnx_inst, builder);
    }
    case ONNXExtOpCode::IDENTITY: {
      return {onnx_inst->GetOperand(0)};
    }
    case ONNXExtOpCode::ONEHOT: {
      return ConvertOneHot(onnx_inst, builder);
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
    // Todo: DNNl and ODLA should support quant/dequant op, for performace
    // considerations; Now split the Hgai quant op to
    // Mul+Add+Round+Cast+Max+Min+Transpose; Split Hgai dequant op to
    // Cast+Mul+Add+Tranpose;
    case ONNXExtOpCode::HGQUANT: {
      return ConvertHgQuant(onnx_inst, builder);
    }
    case ONNXExtOpCode::HGDEQUANT: {
      return ConvertHgDeQuant(onnx_inst, builder);
    }
    case ONNXExtOpCode::HGENGINE: {
      return ConvertHgEngine(onnx_inst, builder);
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
    }
  }
  return changed;
}

} // end namespace halo
