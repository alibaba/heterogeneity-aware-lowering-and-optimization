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

static std::vector<Def> ConvertLoop(const ONNXExtensionInst* ext,
                                    IRBuilder* builder) {
  // TODO(unknown)
  return {};
}

static std::vector<Def> ConvertSqueeze(const ONNXExtensionInst* ext,
                                       IRBuilder* builder) {
  auto input = ext->GetOperand(0);
  const Type& input_type = input.GetType();

  if (!input_type.IsValid()) {
    return {};
  }

  std::vector<int32_t> squeeze_dims;
  HLCHECK(ext->GetNumOfAttributes() <= 1);
  if (ext->GetNumOfAttributes() == 1) {
    const Attribute* attr = ext->GetAttributes()[0].get();
    HLCHECK(attr->GetName() == "axes");
    squeeze_dims = attr->GetValueAsIntegerList();
  }
  std::vector<int32_t> new_dims;
  for (size_t i = 0, e = input_type.GetNumOfDims(); i < e; ++i) {
    auto size = input_type.GetNumOfElementsInDim(i);
    if (size != 1) {
      new_dims.push_back(size);
    } else {
      if (!squeeze_dims.empty() &&
          std::find(squeeze_dims.begin(), squeeze_dims.end(), i) ==
              squeeze_dims.end()) {
        new_dims.push_back(size);
      }
    }
  }
  ConstantBuilder cb(ext->GetParent()->GetParent());
  Constant* c = cb.CreateConstant(
      ext->GetName() + "_squeeze_dims",
      Type{DataType::INT32, {static_cast<int64_t>(new_dims.size())}},
      new_dims.data());
  builder->SetInsertAfter(ext);
  auto new_inst = builder->CreateReshape(ext->GetName(), {input, *c});
  return {*new_inst};
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
  auto op0 = ext->GetOperand(0);
  auto op1 = ext->GetOperand(1); // starts
  auto op2 = ext->GetOperand(2); // ends

  auto& input_type = op0.GetType();
  if (!input_type.IsValid()) {
    return {};
  }
  if (!IsA<Constant>(op1) || !IsA<Constant>(op2)) {
    return {};
  }

  builder->SetInsertAfter(ext);
  int input_dims = input_type.GetNumOfDims();

  Constant* c_starts = DynCast<Constant>(op1);
  const auto& starts_type = op1.GetType();
  Constant* c_ends = DynCast<Constant>(op2);
  const auto& ends_type = op2.GetType();

  HLCHECK(ends_type.GetNumOfDims() == starts_type.GetNumOfDims());

  std::unordered_set<int32_t> axes;

  ConstantBuilder cb(ext->GetParent()->GetParent());
  // If no axes operand, assumes all axes are sliced and steps are 1.
  Def op3 = Def::GetUndefined();
  Def op4 = Def::GetUndefined();

  if ((ext->GetNumOfOperands() == 3) || (ext->GetNumOfOperands() == 4)) {
    std::vector<int> steps(input_dims, 1);
    Constant* c_steps =
        cb.CreateConstant(ext->GetName() + "_steps",
                          Type{DataType::INT32, {input_dims}}, steps.data());
    op4 = *c_steps;
    if (ext->GetNumOfOperands() == 3) {
      std::vector<int> data(input_dims);
      for (int i = 0; i < input_dims; ++i) {
        axes.insert(i);
        data[i] = i;
      }
      Constant* c_axes =
          cb.CreateConstant(ext->GetName() + "_axes",
                            Type{DataType::INT32, {input_dims}}, data.data());
      op3 = *c_axes;
    }
  }

  if (ext->GetNumOfOperands() >= 4) {
    op3 = ext->GetOperand(3); // axes
    if (!IsA<Constant>(op3)) {
      return {};
    }

    if (ext->GetNumOfOperands() > 4) {
      op4 = ext->GetOperand(4); // steps
      if (!IsA<Constant>(op4)) {
        return {};
      }
    }

    Constant* c_axes = DynCast<Constant>(op3);
    auto e_a = op3.GetType().GetTotalNumOfElements();
    for (int i = 0, tmp = 0; i != e_a; ++i) {
      if (op3.GetType().GetDataType() == DataType::INT32) {
        tmp = c_axes->GetData<int32_t>(i);
      } else if (op3.GetType().GetDataType() == DataType::INT64) {
        tmp = static_cast<int32_t>(c_axes->GetData<int64_t>(i));
      }
      axes.insert(tmp >= 0 ? tmp : tmp + input_dims);
    }

    HLCHECK(!axes.empty());
  }

  auto e_s = op1.GetType().GetTotalNumOfElements();
  int32_t start = 0;
  std::vector<int32_t> starts;
  starts.reserve(axes.size());
  for (int i = 0, j = 0; i < input_dims; ++i) {
    if (axes.count(i) != 0) {
      if (starts_type.GetDataType() == DataType::INT32) {
        start = c_starts->GetData<int32_t>(j);
      } else if (starts_type.GetDataType() == DataType::INT64) {
        start = static_cast<int32_t>(c_starts->GetData<int64_t>(j));
      }

      if (start < 0) {
        start = input_type.GetNumOfElementsInDim(i) - start;
      } else if (start > input_type.GetNumOfElementsInDim(i)) {
        start = input_type.GetNumOfElementsInDim(i);
      }
      starts.push_back(start);
      j++;
    }
  }

  auto e_d = op2.GetType().GetTotalNumOfElements();
  std::vector<int32_t> ends;
  ends.reserve(axes.size());
  int32_t end = 0;
  for (int i = 0, j = 0; i < input_dims; ++i) {
    if (axes.count(i) != 0) {
      if (ends_type.GetDataType() == DataType::INT32) {
        auto tmp = c_ends->GetData<int32_t>(j);
        if (tmp == std::numeric_limits<int32_t>::max()) {
          // INT_MAX represents end dimension is unknown,
          // thus take all elements
          tmp = input_type.GetNumOfElementsInDim(i);
        }
        end = tmp;
      } else if (ends_type.GetDataType() == DataType::INT64) {
        auto tmp = c_ends->GetData<int64_t>(j);
        if (tmp == std::numeric_limits<int64_t>::max()) {
          // INT64_MAX represents end dimension is unknown,
          // thus take all elements
          tmp = input_type.GetNumOfElementsInDim(i);
        }
        end = static_cast<int32_t>(tmp);
      }

      if (end < 0) {
        end = input_type.GetNumOfElementsInDim(i) - end;
      } else if (end > input_type.GetNumOfElementsInDim(i)) {
        end = input_type.GetNumOfElementsInDim(i);
      }
      ends.push_back(end);
      j++;
    }
  }

  // calculate sizes = ends - starts
  std::transform(ends.begin(), ends.end(), starts.begin(), ends.begin(),
                 std::minus<int32_t>());

  Constant* c_begins_norm = cb.CreateConstant(
      ext->GetName() + "_starts",
      Type{DataType::INT32, {static_cast<int64_t>(e_s)}}, starts.data());

  Constant* c_sizes_norm = cb.CreateConstant(
      ext->GetName() + "_sizes",
      Type{DataType::INT32, {static_cast<int64_t>(e_d)}}, ends.data());

  SliceInst* slice = builder->CreateSlice(
      ext->GetName(), {op0, *c_begins_norm, *c_sizes_norm, op4, op3});
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

  // if idx_shape[i] and input_shape[i] are 1 for all dims except for "axis", it
  // can be converted to Gather. Otherwise, if idx_shape is a result of
  // broadcasting, the input of broadcasting might be converted.
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
        idx_op.GetDef()->GetName() + "_shape",
        Type{DataType::INT64, {static_cast<int64_t>(new_dims.size())}},
        new_dims.data());

    builder->SetInsertAfter(ext);
    auto reshape = builder->CreateReshape(
        idx_op.GetDef()->GetName() + "_reshape", idx_op, *c);
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
    case ONNXExtOpCode::LOOP: {
      return ConvertLoop(onnx_inst, builder);
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
    }
  }
  return changed;
}

} // end namespace halo
