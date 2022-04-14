//===- tfextension_legalizer.cc -------------------------------------------===//
//
// Copyright (C) 2019-2021 Alibaba Group Holding Limited.
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

#include "halo/lib/transforms/tfextension_legalizer.h"

#include <cmath>
#include <numeric>

#include "halo/api/halo_data.h"
#include "halo/lib/framework/common.h"
#include "halo/lib/framework/data_layout.h"
#include "halo/lib/ir/common_instructions.h"
#include "halo/lib/ir/extension_instructions.h"
#include "halo/lib/ir/ir_builder.h"
#include "halo/lib/transforms/transforms_util.h"

namespace halo {

static std::vector<Def> ConvertAddN(const TFExtensionInst* ext,
                                    IRBuilder* builder) {
  HLCHECK(ext->GetNumOfOperands() >= 2 && "Invalid AddN");
  builder->SetInsertAfter(ext);
  Instruction* acc = builder->CreateAdd(ext->GetName(), ext->GetOperand(0),
                                        ext->GetOperand(1));
  for (int i = 2, e = ext->GetNumOfOperands(); i != e; ++i) {
    acc = builder->CreateAdd(ext->GetName() + "_" + std::to_string(i - 2), *acc,
                             ext->GetOperand(i));
  }
  return {*acc};
}

static std::vector<Def> ConvertBroadcastTo(const TFExtensionInst* ext,
                                           IRBuilder* builder) {
  // Assume the consumer supports implicit broadcasting.
  return {ext->GetOperand(0)};
}

static std::vector<Def> ConvertReshape(const TFExtensionInst* tf_reshape,
                                       IRBuilder* builder) {
  HLCHECK(tf_reshape->GetNumOfOperands() > 0 &&
          tf_reshape->GetNumOfOperands() <= 2);
  Instruction* new_inst = nullptr;

  builder->SetInsertAfter(tf_reshape);
  if (tf_reshape->GetNumOfOperands() == 2) {
    new_inst = builder->CreateReshape(
        tf_reshape->GetName(),
        {tf_reshape->GetOperand(0), tf_reshape->GetOperand(1)});
  } else {
    // Convert attribute to constant operand.
    auto shape = FindAttributeValue<std::vector<int>>(*tf_reshape, "shape", {});
    HLCHECK(!shape.empty());
    ConstantBuilder cb(tf_reshape->GetParent()->GetParent());
    Constant* c = cb.CreateConstant(
        tf_reshape->GetName() + "_shape",
        Type{DataType::INT32, {static_cast<int64_t>(shape.size())}},
        shape.data());
    new_inst = builder->CreateReshape(tf_reshape->GetName(),
                                      {tf_reshape->GetOperand(0), *c});
  }
  return {*new_inst};
}

static std::vector<Def> ConvertSquare(const TFExtensionInst* tf_inst,
                                      IRBuilder* builder) {
  builder->SetInsertAfter(tf_inst);
  const auto& op = tf_inst->GetOperand(0);
  auto new_inst = builder->CreateMul(tf_inst->GetName(), op, op);
  return {*new_inst};
}

static std::vector<Def> ConvertSqueeze(const TFExtensionInst* tf_squeeze,
                                       IRBuilder* builder) {
  auto input = tf_squeeze->GetOperand(0);
  const Type& input_type = input.GetType();

  if (!input_type.IsValid()) {
    return {};
  }

  std::vector<int32_t> squeeze_dims;
  HLCHECK(tf_squeeze->GetNumOfAttributes() <= 1);
  if (tf_squeeze->GetNumOfAttributes() == 1) {
    squeeze_dims =
        FindAttributeValue(*tf_squeeze, "squeeze_dims", squeeze_dims);
    HLCHECK(!squeeze_dims.empty());
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
  ConstantBuilder cb(tf_squeeze->GetParent()->GetParent());
  Constant* c = cb.CreateConstant(
      tf_squeeze->GetName() + "_squeeze_dims",
      Type{DataType::INT32, {static_cast<int64_t>(new_dims.size())}},
      new_dims.data());
  builder->SetInsertAfter(tf_squeeze);
  auto new_inst = builder->CreateReshape(tf_squeeze->GetName(), {input, *c});
  return {*new_inst};
}

static std::vector<Def> ConvertCast(const TFExtensionInst* ext,
                                    IRBuilder* builder) {
  HLCHECK(ext->GetNumOfAttributes() == 3);
  const Attribute* attr = ext->GetAttributes()[0].get();
  HLCHECK(attr->GetName() == "SrcT");
  const auto& src_type = attr->GetValueAsEnumDataType();
  attr = ext->GetAttributes()[1].get();
  HLCHECK(attr->GetName() == "DstT");
  const auto& dst_type = attr->GetValueAsEnumDataType();
  attr = ext->GetAttributes()[2].get();
  HLCHECK(attr->GetName() == "Truncate");
  // bool truncate = attr->GetValueAsBool();
  builder->SetInsertAfter(ext);
  auto op0 = ext->GetOperand(0);
  if (Type::IsIntegerType(src_type)) {
    if (Type::IsIntegerType(dst_type)) {
      ZExtInst* new_inst = builder->CreateZExt(ext->GetName(), op0);
      new_inst->SetDataType(dst_type);
      return {*new_inst};
    }
    HLCHECK(Type::IsFloatingPointType(dst_type));
    SItoFPInst* new_inst = builder->CreateSItoFP(ext->GetName(), op0);
    new_inst->SetDataType(dst_type);
    return {*new_inst};
  }
  if (Type::IsFloatingPointType(src_type)) {
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
  }
  return {};
}

static std::vector<Def> ConvertExpandDims(const TFExtensionInst* ext,
                                          IRBuilder* builder) {
  HLCHECK(ext->GetNumOfOperands() == 2);
  auto input = ext->GetOperand(0);
  auto index = ext->GetOperand(1);
  const Type& input_type = input.GetType();

  const Constant* axis_c = DynCast<Constant>(index);
  if (!input_type.IsValid() || axis_c == nullptr) {
    return {};
  }

  int64_t axis = axis_c->GetDataAsInt64(0);
  int input_rank = input_type.GetNumOfDims();
  HLCHECK(-1 - input_rank <= axis && axis <= input_rank);

  std::vector<int64_t> new_dims(input_type.GetDimSizes());
  if (axis < 0) {
    //  if 't' is a tensor of shape [2], expand_dims(t, -1) ==> [2, 1]
    axis += input_rank + 1;
  }
  new_dims.insert(new_dims.begin() + axis, 1);

  ConstantBuilder cb(ext->GetParent()->GetParent());
  Constant* c = cb.CreateConstant(
      ext->GetName() + "_expand_dims",
      Type{DataType::INT64, {static_cast<int64_t>(new_dims.size())}},
      new_dims.data());
  builder->SetInsertAfter(ext);
  auto new_inst = builder->CreateReshape(ext->GetName(), {input, *c});
  return {*new_inst};
}

static std::vector<Def> ConvertFill(const TFExtensionInst* ext,
                                    IRBuilder* builder) {
  HLCHECK(ext->GetNumOfOperands() == 2);
  auto dims = ext->GetOperand(0);
  auto value = ext->GetOperand(1);
  if (!IsA<Constant>(value)) {
    return {};
  }
  if (!IsA<Constant>(dims)) {
    auto& dims_type = dims.GetType();
    auto& value_type = value.GetType();
    if (dims_type.IsValid()) {
      builder->SetInsertAfter(ext);
      ConstantBuilder cb(ext->GetParent()->GetParent());

      Constant* value_c = DynCast<Constant>(value.GetOwner());
      DataType value_dt =
          FindAttributeValue<DataType>(*ext, "dtype", DataType::INVALID);
      value_dt =
          (value_dt == DataType::INVALID) ? value_type.GetDataType() : value_dt;

      auto rank = dims_type.GetTotalNumOfElements();
      std::vector<int64_t> constant_dims(rank, 1);

      Type new_type(value_dt, constant_dims);
      Constant* new_value_c = nullptr;

      switch (value_dt) {
        case DataType::INT32: {
          std::vector<int32_t> data(1, value_c->GetData<int32_t>(0));
          new_value_c = cb.CreateConstant(ext->GetName() + "_value_constant",
                                          new_type, data.data());
          break;
        }
        case DataType::FLOAT32: {
          std::vector<float> data(1, value_c->GetData<float>(0));
          new_value_c = cb.CreateConstant(ext->GetName() + "_value_constant",
                                          new_type, data.data());
          break;
        }
        case DataType::INT64: {
          std::vector<int64_t> data(1, value_c->GetData<int64_t>(0));
          new_value_c = cb.CreateConstant(ext->GetName() + "_value_constant",
                                          new_type, data.data());
          break;
        }
        default:
          HLCHECK(0 && "Unimplemented data type.");
      }

      auto new_inst =
          builder->CreateTileDynamic(ext->GetName(), *new_value_c, dims);
      return {*new_inst};
    }
    return {};
  }
  std::vector<int64_t> shape;
  Constant* dims_c = DynCast<Constant>(dims);
  const Type& dims_type = dims.GetType();
  for (size_t i = 0, e = dims_type.GetTotalNumOfElements(); i < e; ++i) {
    shape.push_back(dims_c->GetData<int32_t>(i));
  }
  DataType value_dt = value.GetType().GetDataType();
  Type new_type{value_dt, shape};
  size_t data_size = new_type.GetTotalNumOfElements();
  Constant* value_c = DynCast<Constant>(value.GetOwner());
  ConstantBuilder cb(ext->GetParent()->GetParent());
  Constant* c = nullptr;
  switch (value_dt) {
    case DataType::INT32: {
      std::vector<int32_t> data(data_size, value_c->GetData<int32_t>(0));
      c = cb.CreateConstant(ext->GetName(), new_type, data.data());
      break;
    }
    case DataType::FLOAT32: {
      std::vector<float> data(data_size, value_c->GetData<float>(0));
      c = cb.CreateConstant(ext->GetName(), new_type, data.data());
      break;
    }
    case DataType::INT64: {
      std::vector<int64_t> data(data_size, value_c->GetData<int64_t>(0));
      c = cb.CreateConstant(ext->GetName(), new_type, data.data());
      break;
    }
    case DataType::FLOAT16: {
      const void* raw_ptr = value_c->GetRawDataPtr();
      std::vector<int16_t> data(
          data_size, *static_cast<const uint16_t*>(raw_ptr)); // NOLINT
      c = cb.CreateConstant(ext->GetName(), new_type, data.data());
      break;
    }
    default:
      HLCHECK(0 && "Unimplemented data type.");
  }
  if (c != nullptr) {
    return {*c};
  }
  return {};
}

static std::vector<Def> ConvertSize(const TFExtensionInst* ext,
                                    IRBuilder* builder) {
  const auto& type = ext->GetOperand(0).GetType();
  if (!type.IsValid()) {
    return {};
  }
  auto n = type.GetTotalNumOfElements();
  ConstantBuilder cb(ext->GetParent()->GetParent());
  return {*(cb.CreateConstant(ext->GetName(), Type{DataType::INT64, {1}}, &n))};
}

static std::vector<Def> ConvertSplit(const TFExtensionInst* ext,
                                     IRBuilder* builder) {
  auto input = ext->GetOperand(1);
  auto split_dim = ext->GetOperand(0);
  auto num_split = FindAttributeValue<int>(*ext, "num_split", 0);
  const Type& input_type = input.GetType();
  if (!input_type.IsValid()) {
    return {};
  }
  int rank = input_type.GetNumOfDims();
  const Constant* split_dim_c = DynCast<Constant>(split_dim);
  HLCHECK(split_dim_c != nullptr && "split_dim is not a constant");
  HLCHECK(split_dim_c->GetResultType().IsScalar() &&
          "split_dim is not a scalar");
  int axis = split_dim_c->GetDataAsInt64(0);
  HLCHECK(axis >= 0 && axis < rank && "Invalid split dim");
  auto orig_size = input_type.GetNumOfElementsInDim(axis);
  HLCHECK(num_split > 0 && (orig_size % num_split == 0) && "Invalid num_split");
  int len = static_cast<int>(orig_size / num_split);
  builder->SetInsertAfter(ext);
  ConstantBuilder cb(ext->GetParent()->GetParent());
  const int32_t step = 1;
  const Type param_type{DataType::INT32, {1}};
  auto c_len = cb.CreateConstant(ext->GetName() + "_len", param_type, &len);
  auto c_step = cb.CreateConstant(ext->GetName() + "_step", param_type, &step);
  auto c_axis = cb.CreateConstant(ext->GetName() + "_axis", param_type, &axis);
  std::vector<Def> ret;
  ret.reserve(num_split);
  for (int i = 0, start = 0; i < num_split; ++i) {
    auto c_start = cb.CreateConstant(
        ext->GetName() + "_start_" + std::to_string(i), param_type, &start);
    auto slice =
        builder->CreateSlice(ext->GetName() + "_" + std::to_string(i), input,
                             *c_start, *c_len, *c_step, *c_axis);
    ret.push_back(*slice);
    start += len;
  }
  return ret;
}

static std::vector<Def> ConvertSplitToSplit(const TFExtensionInst* ext,
                                            IRBuilder* builder) {
  auto input = ext->GetOperand(1);
  auto split_dim = ext->GetOperand(0);
  auto num_split = FindAttributeValue<int>(*ext, "num_split", 0);
  const Type& input_type = input.GetType();
  if (!input_type.IsValid()) {
    return {};
  }
  builder->SetInsertAfter(ext);
  auto split_inst = builder->CreateSplit(ext->GetName(), split_dim, input);
  split_inst->SetNumSplit(num_split);
  split_inst->SetNumOfResults(num_split);
  std::vector<Def> ret;
  ret.reserve(num_split);
  for (int i = 0; i < num_split; ++i) {
    ret.push_back(Def(split_inst, i));
  }
  return ret;
}

static std::vector<Def> ConvertStridedSlice(const TFExtensionInst* ext,
                                            IRBuilder* builder) {
  HLCHECK(ext->GetNumOfOperands() >= 4);
  auto input = ext->GetOperand(0);
  auto begin = ext->GetOperand(1);
  auto end = ext->GetOperand(2);
  auto strides = ext->GetOperand(3);
  const auto& input_type = input.GetType();

  if (!input_type.IsValid() || !IsA<Constant>(begin) || !IsA<Constant>(end) ||
      !IsA<Constant>(strides)) {
    return {};
  }

  int begin_mask = ext->GetAttributes()[0]->GetValueAsInteger();
  int end_mask = ext->GetAttributes()[1]->GetValueAsInteger();
  int ellipsis_mask = ext->GetAttributes()[2]->GetValueAsInteger();
  int shrink_mask = ext->GetAttributes()[4]->GetValueAsInteger();
  int new_axis_mask = ext->GetAttributes()[3]->GetValueAsInteger();

  size_t n = begin.GetType().GetTotalNumOfElements();
  auto begin_c = DynCast<Constant>(begin.GetOwner());
  auto end_c = DynCast<Constant>(end.GetOwner());
  auto strides_c = DynCast<Constant>(strides.GetOwner());
  ConstantBuilder cb(ext->GetParent()->GetParent());
  // constant folding
  if (IsA<Constant>(input) && input_type.GetNumOfDims() == 1) {
    HLCHECK((ellipsis_mask == 0) && "Not supported ellipsis mask value.");
    int32_t begin_i = begin_mask == 1 ? 0 : begin_c->GetData<int32_t>(0);
    int32_t end_i = end_mask == 1 || end_c->GetData<int32_t>(0) == -1
                        ? input_type.GetNumOfElementsInDim(0)
                        : end_c->GetData<int32_t>(0);
    int32_t strides_i = strides_c->GetData<int32_t>(0);
    Constant* c_input = DynCast<Constant>(input.GetOwner());
    unsigned char* data_ptr =
        static_cast<unsigned char*>(c_input->GetRawDataPtr());
    DataType dt = input_type.GetDataType();
    HLCHECK(end_i >= begin_i);
    size_t num_elements = (end_i - begin_i) / strides_i;
    std::vector<int64_t> result_shape;
    if (shrink_mask != 0) {
      HLCHECK(shrink_mask == 1);
    } else {
      result_shape.push_back(num_elements);
    }

    auto copy_size = c_input->GetElementSizeInBytes();
    auto strides = strides_i * copy_size;
    std::vector<char> result(num_elements * copy_size);
    for (size_t i = 0, offset = begin_i * copy_size; i < num_elements; i++) {
      memcpy(&result[copy_size * i], data_ptr + offset, copy_size); // NOLINT
      offset += strides;
    }
    Constant* c_result =
        cb.CreateConstant(ext->GetName() + "_constant_folding",
                          Type{dt, result_shape}, result.data());
    return {*c_result};
  }

  // General extension handling: --> slice + reshape.
  std::vector<int32_t> new_begin;
  std::vector<int32_t> new_size;
  bool empty_slice = false;
  for (size_t i = 0, ellipsis_cnt = 0; i < n; ++i) {
    int32_t begin_i = begin_c->GetData<int32_t>(i);
    int32_t end_i = end_c->GetData<int32_t>(i);
    int32_t strides_i = strides_c->GetData<int32_t>(i);
    int32_t dims_i = input.GetType().GetNumOfElementsInDim(i);
    auto index = 1 << i;
    bool dynamic_stridedslice_i = false;

    if (end_i < 0) {
      end_i += dims_i;
    }

    if ((ellipsis_mask & index) != 0) {
      HLCHECK(new_axis_mask == 0 && shrink_mask == 0 && "Unhandled ellipsis");
      ellipsis_cnt = 1 + n - input_type.GetNumOfDims();
    }
    if ((begin_mask & index) != 0 || ellipsis_cnt != 0) {
      new_begin.push_back(0);
    } else {
      new_begin.push_back(begin_i);
    }
    if ((shrink_mask & index) != 0) {
      new_size.push_back(1);
    } else if ((end_mask & index) != 0 || ellipsis_cnt != 0) {
      if ((dims_i == -1) && (new_begin.back() == 0) && (strides_i == 1)) {
        dynamic_stridedslice_i = true;
      }
      new_size.push_back((dims_i - new_begin.back()) / strides_i);
    } else {
      new_size.push_back((end_i - new_begin.back()) / strides_i);
    }
    if (ellipsis_cnt > 0) {
      --ellipsis_cnt;
    }
    HLCHECK(new_size.back() >= 0 ||
            dynamic_stridedslice_i); // TF allows empty tensor
    if (new_size.back() == 0) {
      empty_slice = true;
    }
  }
  if (empty_slice) {
    std::vector<int64_t> new_shape(new_size.begin(), new_size.end());
    Constant* c_empty =
        cb.CreateConstant(ext->GetName() + "_empty",
                          Type{input_type.GetDataType(), new_shape}, nullptr);
    return {*c_empty};
  }

  Constant* c_begin = cb.CreateConstant(
      ext->GetName() + "_new_begin",
      Type{DataType::INT32, {static_cast<int64_t>(n)}}, new_begin.data());
  Constant* c_end = cb.CreateConstant(
      ext->GetName() + "_new_size",
      Type{DataType::INT32, {static_cast<int64_t>(n)}}, new_size.data());
  builder->SetInsertAfter(ext);
  IRObject* new_slice_inst =
      builder->CreateSlice(ext->GetName() + "_slice",
                           {ext->GetOperand(0), *c_begin, *c_end, strides});

  // dim expansion or shrink
  if (new_axis_mask != 0 || shrink_mask != 0) {
    std::vector<int32_t> new_dims;
    for (size_t i = 0; i != n; ++i) {
      if ((shrink_mask & (1 << i)) != 0) {
        HLCHECK(new_size.at(i) == 1);
      } else {
        new_dims.push_back(new_size.at(i));
      }
    }
    for (size_t i = 0; i != n; ++i) {
      if ((new_axis_mask & (1 << i)) != 0) {
        new_dims.insert(new_dims.begin() + i, 1);
      }
      new_axis_mask >>= 1;
    }
    std::vector<int64_t> new_shape;
    if (new_dims.empty()) {
      new_dims.push_back(1);
    } else {
      new_shape.push_back(static_cast<int64_t>(new_dims.size()));
    }
    Constant* c_shape =
        cb.CreateConstant(new_slice_inst->GetName() + "_shape",
                          Type{DataType::INT32, new_shape}, new_dims.data());
    new_slice_inst = builder->CreateReshape(ext->GetName() + "_reshape",
                                            {Def{new_slice_inst, 0}, *c_shape});
  }
  return {*new_slice_inst};
}

static std::vector<Def> ConvertSwitch(const TFExtensionInst* ext,
                                      IRBuilder* builder) {
  const auto& data = ext->GetOperand(0);
  const auto& cond = ext->GetOperand(1);
#if 0
  if (const Constant* pred = DynCast<Constant>(cond);
      pred != nullptr) {
    HLCHECK(pred->GetResultType().GetTotalNumOfElements() == 1);
    bool cond = pred->GetDataAsInt64(0) != 0;
    std::vector<Def> ret_true{Def::GetUndefined(), data};
    std::vector<Def> ret_false{data, Def::GetUndefined()};
    return cond ? ret_true : ret_false;
  }
#endif
// TODO(unknown): move to separate pass?
#if 1
  builder->SetInsertAfter(ext);
  BasicBlockBuilder bb_builder(ext->GetParent()->GetParent());
  const auto& name = ext->GetName();
  auto if_inst = builder->CreateIf(ext->GetName(), cond);
  if_inst->AddOneOperand(data);

  BasicBlock* bb_t = bb_builder.CreateBasicBlock(name + "_true");
  if_inst->SetThenBranch(bb_t);
  IRBuilder builder_t(bb_t);
  auto arg_builder_t = std::make_unique<ArgumentBuilder>(bb_t);
  auto arg_t = arg_builder_t->CreateArgument(name + "_t", data.GetType());
  builder_t.CreateReturn(name + "ret_t", *arg_t);

  BasicBlock* bb_f = bb_builder.CreateBasicBlock(name + "_false");
  IRBuilder builder_f(bb_f);
  if_inst->SetElseBranch(bb_f);
  auto arg_builder_f = std::make_unique<ArgumentBuilder>(bb_f);
  auto arg_f = arg_builder_f->CreateArgument(name + "_f", data.GetType());
  builder_f.CreateReturn(name + "ret_f", *arg_f);
  if_inst->SetNumOfResults(2);
  return {Def(if_inst, 0), Def(if_inst, 1)};
#else
  return {};
#endif
}

static std::vector<Def> ConvertMerge(const TFExtensionInst* ext,
                                     IRBuilder* builder) {
  Def ret = Def::GetUndefined();
  for (int i = 0, e = ext->GetNumOfOperands(); i != e; ++i) {
    const auto& op = ext->GetOperand(i);
    // Check if there is one and only one valid operand.
    if (!op.IsNull()) {
      if (!ret.IsNull()) {
        return {};
      }
      ret = op;
    }
  }
  if (!ret.IsNull()) {
    return {ret};
  }
  return {};
}

static std::vector<Def> ConvertZerosLike(const TFExtensionInst* ext,
                                         IRBuilder* builder) {
  const auto& op0 = ext->GetOperand(0);
  const auto& op0_type = op0.GetType();
  if (!op0_type.IsValid()) {
    return {};
  }
  if (!op0_type.IsStaticShape()) {
    ConstantBuilder cb(ext->GetParent()->GetParent());
    builder->SetInsertAfter(ext);
    auto zeroslike_shape_inst =
        builder->CreateShape(ext->GetName() + "_shape", op0);

    DataType value_dt =
        FindAttributeValue<DataType>(*ext, "dtype", DataType::INVALID);
    value_dt =
        (value_dt == DataType::INVALID) ? op0_type.GetDataType() : value_dt;
    auto zeros_rank = op0_type.GetNumOfDims();
    std::vector<int64_t> zeros_constant_dims(zeros_rank, 1);
    Type new_type(value_dt, zeros_constant_dims);
    Constant* zeros_constant = nullptr;
    switch (value_dt) {
      case DataType::INT32: {
        std::vector<int32_t> data(1);
        zeros_constant = cb.CreateConstant(ext->GetName() + "_zero_constant",
                                           new_type, data.data());
        break;
      }
      case DataType::FLOAT32: {
        std::vector<float> data(1);
        zeros_constant = cb.CreateConstant(ext->GetName() + "_zero_constant",
                                           new_type, data.data());
        break;
      }
      case DataType::INT64: {
        std::vector<int64_t> data(1);
        zeros_constant = cb.CreateConstant(ext->GetName() + "_zero_constant",
                                           new_type, data.data());
        break;
      }
      default:
        HLCHECK(0 && "Unimplemented data type.");
    }

    auto zeros_like_inst = builder->CreateTileDynamic(
        ext->GetName(), *zeros_constant, *zeroslike_shape_inst);
    return {*zeros_like_inst};
  }
  DataType vt = FindAttributeValue<DataType>(*ext, "dtype", DataType::INVALID);
  vt = (vt == DataType::INVALID) ? op0_type.GetDataType() : vt;
  ConstantBuilder cb(ext->GetParent()->GetParent());
  DefaultDataLayout dl;
  std::vector<char> buf(dl.DataLayout::Bytes(op0_type));
  auto c = cb.CreateConstant(ext->GetName(), Type{vt, op0_type.GetDimSizes()},
                             buf.data());
  return {*c};
}

template <typename T>
static std::vector<T> GetConstantVals(const Constant& c) {
  const T* ptr = static_cast<const T*>(c.GetRawDataPtr());
  size_t n = c.GetResultType().GetTotalNumOfElements();
  std::vector<T> ret{ptr, ptr + n}; // NOLINT
  return ret;
}

static std::tuple<std::vector<int32_t>, std::vector<int32_t>>
ValidateBatchSpace(const TFExtensionInst* tf_inst) {
  HLCHECK(tf_inst->GetNumOfOperands() == 3);
  auto op0 = tf_inst->GetOperand(0);
  const auto& input_type = op0.GetType();
  const int64_t input_dims = input_type.GetNumOfDims();
  HLCHECK(input_dims >= 2); // batch + space + remaining
  const auto& op1 = tf_inst->GetOperand(1);
  const auto& op2 = tf_inst->GetOperand(2);
  HLCHECK(IsA<Constant>(op1));
  HLCHECK(IsA<Constant>(op2));
  const Constant* block_shape = DynCast<Constant>(op1.GetOwner());
  const Constant* crop_pad = DynCast<Constant>(op2.GetOwner());

  const auto& block_type = block_shape->GetResultType();
  HLCHECK(block_type.GetNumOfDims() == 1);
  unsigned space_n = block_type.GetNumOfElementsInDim(0);
  HLCHECK(block_type.GetDataType() == DataType::INT32);
  const auto& crop_pad_type = crop_pad->GetResultType();
  HLCHECK(crop_pad_type.GetNumOfDims() == 2);
  HLCHECK(crop_pad_type.GetNumOfElementsInDim(0) == space_n);
  HLCHECK(crop_pad_type.GetNumOfElementsInDim(1) == 2);
  HLCHECK(crop_pad_type.GetDataType() == DataType::INT32);
  HLCHECK(input_dims >= space_n + 1);

  auto blocking_vals = GetConstantVals<int32_t>(*block_shape);
  auto crop_pad_vals = GetConstantVals<int32_t>(*crop_pad);
  return make_tuple(blocking_vals, crop_pad_vals);
}

static std::vector<Def> ConvertBatch2Space(const TFExtensionInst* tf_inst,
                                           IRBuilder* builder) {
  // Expand to reshape->permute->reshape->crop
  const auto& op0 = tf_inst->GetOperand(0);

  const auto& input_type = op0.GetType();
  if (!input_type.IsValid()) {
    return {};
  }

  std::vector<int32_t> blocking_vals;
  std::vector<int32_t> crop_vals;

  std::tie(blocking_vals, crop_vals) = ValidateBatchSpace(tf_inst);
  HLCHECK(!blocking_vals.empty() && !crop_vals.empty());

  const int64_t input_dims = input_type.GetNumOfDims();
  const auto& input_shape = input_type.GetDimSizes();
  size_t space_n = blocking_vals.size();
  auto rem_dims = input_dims - 1 - space_n;

  const std::string& name = tf_inst->GetName();
  ConstantBuilder cb(tf_inst->GetParent()->GetParent());
  builder->SetInsertAfter(tf_inst);

  // First reshape.
  std::vector<int64_t> shape0;
  shape0.reserve(1 + space_n * 2 + rem_dims);
  shape0.insert(shape0.begin(), blocking_vals.begin(), blocking_vals.end());
  int64_t batch = input_shape[0];
  int t = std::accumulate(blocking_vals.begin(), blocking_vals.end(), 1,
                          std::multiplies<int32_t>());
  HLCHECK(batch % t == 0);
  shape0.push_back(batch / t);
  shape0.insert(shape0.end(), input_shape.begin() + 1, input_shape.end());

  Constant* shape0_c = cb.CreateConstant(
      name + "_shape0",
      Type{DataType::INT64,
           std::vector<int64_t>{static_cast<int64_t>(shape0.size())}},
      shape0.data());

  auto reshape0 = builder->CreateReshape(name + "_reshape_0", {op0, *shape0_c});

  // Do permute.
  TransposeInst* perm = builder->CreateTranspose(name + "_perm", {*reshape0});
  std::vector<int32_t> perms;

  perms.reserve(space_n * 2 + 1 + rem_dims);
  perms.push_back(space_n);
  for (unsigned k = 0; k < space_n; ++k) {
    perms.push_back(1 + space_n + k);
    perms.push_back(k);
  }
  for (unsigned i = 0; i < rem_dims; ++i) {
    perms.push_back(1 + space_n * 2 + i);
  }
  perm->SetPermutation(perms);
  auto perm_shape = shape0;
  for (int i = 0, e = shape0.size(); i < e; ++i) {
    perm_shape[i] = shape0[perms[i]];
  }
  perm->GetResultsTypes()[0] = Type{input_type.GetDataType(), perm_shape};

  // Do second reshape.
  std::vector<int64_t> shape1;
  shape1.reserve(input_shape.size());
  shape1.push_back(perm_shape[0]);
  for (unsigned i = 0; i < space_n; ++i) {
    shape1.push_back(perm_shape[1 + i * 2] * perm_shape[1 + i * 2 + 1]);
  }
  shape1.insert(shape1.end(), perm_shape.begin() + 1 + space_n * 2,
                perm_shape.end());
  Constant* shape1_c = cb.CreateConstant(
      name + "_shape1",
      Type{DataType::INT64,
           std::vector<int64_t>{static_cast<int64_t>(shape1.size())}},
      shape1.data());
  auto reshape1 =
      builder->CreateReshape(name + "_reshape_1", {*perm, *shape1_c});

  // Do crop
  bool ignore_crops =
      std::accumulate(crop_vals.begin(), crop_vals.end(), 0) == 0;
  if (ignore_crops) {
    return {*reshape1};
  }

  std::vector<int32_t> start(input_dims);
  std::vector<int32_t> lens(input_dims);
  std::vector<int32_t> strides(input_dims, 1);

  start[0] = 0;
  lens[0] = shape1[0];
  for (size_t i = 0; i < space_n; ++i) {
    start[i + 1] = crop_vals[i * 2];
    lens[i + 1] = shape1[i + 1] - crop_vals[i * 2] - crop_vals[i * 2 + 1];
  }
  for (unsigned i = 0; i < rem_dims; ++i) {
    lens[i + 1 + space_n] = shape1[i + 1 + space_n];
  }
  Type ty{DataType::INT32,
          std::vector<int64_t>{static_cast<int64_t>(input_dims)}};
  Constant* start_c =
      cb.CreateConstant(name + "_slice_start", ty, start.data());
  Constant* lens_c = cb.CreateConstant(name + "_slice_size", ty, lens.data());
  Constant* strides_c =
      cb.CreateConstant(name + "_slice_stride", ty, strides.data());

  SliceInst* slice = builder->CreateSlice(
      name + "_slice", {*reshape1, *start_c, *lens_c, *strides_c});

  return {*slice};
}

static std::vector<Def> ConvertSpace2Batch(const TFExtensionInst* tf_inst,
                                           IRBuilder* builder) {
  // Expand to pad->reshape->permute->reshape
  HLCHECK(tf_inst->GetNumOfOperands() == 3);
  auto op0 = tf_inst->GetOperand(0);
  auto input_type = op0.GetType();
  if (!input_type.IsValid()) {
    return {};
  }
  const int64_t input_dims = input_type.GetNumOfDims();
  HLCHECK(input_dims >= 2); // batch + space + remaining
  const auto& op1 = tf_inst->GetOperand(1);
  const auto& op2 = tf_inst->GetOperand(2);
  HLCHECK(IsA<Constant>(op1));
  HLCHECK(IsA<Constant>(op2));
  const Constant* block_shape = DynCast<Constant>(op1);
  const Constant* paddings = DynCast<Constant>(op2);

  const auto& block_type = block_shape->GetResultType();
  HLCHECK(block_type.GetNumOfDims() == 1);
  unsigned space_n = block_type.GetNumOfElementsInDim(0);
  HLCHECK(block_type.GetDataType() == DataType::INT32);
  const auto& padding_type = paddings->GetResultType();
  HLCHECK(padding_type.GetNumOfDims() == 2);
  HLCHECK(padding_type.GetNumOfElementsInDim(0) == space_n);
  HLCHECK(padding_type.GetNumOfElementsInDim(1) == 2);
  HLCHECK(padding_type.GetDataType() == DataType::INT32);
  HLCHECK(input_dims >= space_n + 1);

  auto padding_vals = GetConstantVals<int32_t>(*paddings);
  auto blocking_vals = GetConstantVals<int32_t>(*block_shape);

  bool ignore_padding =
      std::accumulate(padding_vals.begin(), padding_vals.end(), 0) == 0;

  const std::string& name = tf_inst->GetName();
  ConstantBuilder cb(tf_inst->GetParent()->GetParent());
  builder->SetInsertAfter(tf_inst);

  if (!ignore_padding) {
    std::vector<int32_t> padding_amt(2 * input_dims);
    auto result_dims = input_type.GetDimSizes();
    for (unsigned k = 1; k <= space_n; ++k) {
      padding_amt[k * 2] = padding_vals[(k - 1) * 2];
      padding_amt[k * 2 + 1] = padding_vals[(k - 1) * 2 + 1];
      result_dims[k] += padding_amt[k * 2] + padding_amt[k * 2 + 1];
    }
    auto c = cb.CreateConstant(name + "pad_amt",
                               Type{DataType::INT32, {input_dims, 2}},
                               padding_amt.data());
    PadInst* pad = builder->CreatePad(name + "_pad", {op0, *c});
    pad->SetMode(PadMode::CONSTANT);
    Type new_type{input_type.GetDataType(), result_dims};
    pad->GetResultsTypes()[0] = new_type;
    input_type = new_type;
    op0 = *pad;
  }
  auto rem_dims = input_dims - 1 - space_n;

  std::vector<int64_t> shape0;
  shape0.reserve(1 + space_n * 2 + rem_dims);

  const auto& input_shape = input_type.GetDimSizes();
  shape0.push_back(input_shape[0]);
  for (unsigned k = 0; k < space_n; ++k) {
    HLCHECK(input_shape[k + 1] % blocking_vals[k] == 0);
    shape0.push_back(input_shape[k + 1] / blocking_vals[k]);
    shape0.push_back(blocking_vals[k]);
  }
  shape0.insert(shape0.end(), input_shape.begin() + space_n + 1,
                input_shape.end());

  Constant* shape0_c = cb.CreateConstant(
      name + "_shape0",
      Type{DataType::INT64,
           std::vector<int64_t>{static_cast<int64_t>(shape0.size())}},
      shape0.data());

  auto reshape0 = builder->CreateReshape(name + "_reshape_0", {op0, *shape0_c});

  TransposeInst* perm = builder->CreateTranspose(name + "_perm", {*reshape0});
  std::vector<int32_t> perms;
  perms.reserve(shape0.size());
  for (unsigned k = 0; k < space_n; ++k) {
    perms.push_back(2 + k * 2);
  }
  perms.push_back(0);
  for (unsigned k = 0; k < space_n; ++k) {
    perms.push_back(1 + k * 2);
  }
  for (unsigned i = 0; i < rem_dims; ++i) {
    perms.push_back(i + 1 + space_n * 2);
  }
  perm->SetPermutation(perms);

  std::vector<int64_t> output_shape;
  output_shape.reserve(input_dims);
  output_shape.push_back(std::accumulate(blocking_vals.begin(),
                                         blocking_vals.end(), input_shape[0],
                                         std::multiplies<int64_t>()));
  for (unsigned k = 0; k < space_n; ++k) {
    output_shape.push_back(shape0[1 + k * 2]);
  }
  output_shape.insert(output_shape.end(), shape0.begin() + 1 + space_n * 2,
                      shape0.end());

  Constant* shape1_c = cb.CreateConstant(
      name + "_shape1",
      Type{DataType::INT64,
           std::vector<int64_t>{static_cast<int64_t>(output_shape.size())}},
      output_shape.data());
  auto reshape1 =
      builder->CreateReshape(name + "_reshape_1", {*perm, *shape1_c});

  return {*reshape1};
}

static std::vector<Def> GetPlaceholder(const TFExtensionInst* tf_inst,
                                       IRBuilder* builder) {
  builder->SetInsertAfter(tf_inst);
  Function* func = tf_inst->GetParent()->GetParent();
  ArgumentBuilder arg_builder(func);
  Argument* arg = arg_builder.CreateArgument(
      "input",
      Type(DataType::FLOAT32, std::vector<int64_t>{1, 240, 240, 3})); // NOLINT
  return {*arg};
}

static std::vector<Def> ConvertHgQuant(const TFExtensionInst* ext,
                                       IRBuilder* builder) {
  auto input = ext->GetOperand(0);
  const auto& input_type = input.GetType();
  if (!input_type.IsValid()) {
    return {};
  }

  int attr_idx = 0;
  std::vector<float> in_scale =
      ext->GetAttributes()[attr_idx++]->GetValueAsFloatList();
  std::vector<float> in_bias =
      ext->GetAttributes()[attr_idx++]->GetValueAsFloatList();
  std::string qtype = ext->GetAttributes()[attr_idx++]->GetValueAsString();
  bool is_per_channel = ext->GetAttributes()[attr_idx++]->GetValueAsBool();

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
  Type type{DataType::FLOAT32, std::vector<int64_t>{channel_size}};
  if (is_per_channel) {
    HLCHECK(in_scale.size() == static_cast<size_t>(channel_size));
    HLCHECK(in_bias.size() == static_cast<size_t>(channel_size));
    c_scale =
        cb.CreateConstant(ext->GetName() + "_const_scale", type, in_scale);
    c_bias = cb.CreateConstant(ext->GetName() + "_const_bias", type, in_bias);
  } else {
    HLCHECK(in_scale.size() == 1);
    HLCHECK(in_bias.size() == 1);
    std::vector<float> scale_data(channel_size, in_scale[0]);
    std::vector<float> bias_data(channel_size, in_bias[0]);
    c_scale =
        cb.CreateConstant(ext->GetName() + "_const_scale", type, scale_data);
    c_bias = cb.CreateConstant(ext->GetName() + "_const_bias", type, bias_data);
  }

  // convert to mul+bias+round+cast+clip
  // Mul
  input = *(builder->CreateMul(ext->GetName() + "_scale", {input, *c_scale}));
  input = *(builder->CreateAdd(ext->GetName() + "_bias", {input, *c_bias}));
  // round
  input = *(builder->CreateRound(ext->GetName() + "_round", {input}));
  // cast
  DataType dst_type;
  int num_bits = kEightBits;
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

static std::vector<Def> ConvertHgDeQuant(const TFExtensionInst* ext,
                                         IRBuilder* builder) {
  // HLCHECK(0 && "Wrong ConvertHgDeQuant");
  auto input = ext->GetOperand(0);

  const auto& input_type = input.GetType();
  if (!input_type.IsValid()) {
    return {};
  }

  int attr_idx = 0;
  std::vector<float> in_scale =
      ext->GetAttributes()[attr_idx++]->GetValueAsFloatList();
  std::vector<float> in_bias =
      ext->GetAttributes()[attr_idx++]->GetValueAsFloatList();
  bool is_per_channel = ext->GetAttributes()[attr_idx++]->GetValueAsBool();
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
  if (is_per_channel) {
    HLCHECK(in_scale.size() == static_cast<size_t>(channel_size));
    HLCHECK(in_bias.size() == static_cast<size_t>(channel_size));
    c_scale =
        cb.CreateConstant(ext->GetName() + "_const_scale", type, in_scale);
    c_bias = cb.CreateConstant(ext->GetName() + "_const_bias", type, in_bias);
  } else {
    HLCHECK(in_scale.size() == 1);
    HLCHECK(in_bias.size() == 1);
    std::vector<float> scale_data(channel_size, in_scale[0]);
    std::vector<float> bias_data(channel_size, in_bias[0]);
    c_scale =
        cb.CreateConstant(ext->GetName() + "_const_scale", type, scale_data);
    c_bias = cb.CreateConstant(ext->GetName() + "_const_bias", type, bias_data);
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

bool FixUpOneHot(OneHotInst* inst, IRBuilder* builder) {
  auto on_value = inst->GetOperand(2);
  if (on_value.GetType().GetTotalNumOfElements() != 1) {
    return false;
  }
  auto off_value = inst->GetOperand(3);
  builder->SetInsertBefore(inst);
  auto off_on =
      builder->CreateConcat(inst->GetName() + "_off_on", {off_value, on_value});
  std::vector<Def> ops{inst->GetOperand(0), inst->GetOperand(1), *off_on,
                       on_value};
  auto new_inst = builder->CreateOneHot(inst->GetName(), ops);
  inst->ReplaceAllUsesWith({*new_inst});
  return true;
}
static std::vector<Def> ConvertTFSplit(const TFExtensionInst* tf_inst,
                                       IRBuilder* builder) {
  return ConvertSplitToSplit(tf_inst, builder);
}
static std::vector<Def> ConvertTFExtension(const TFExtensionInst* tf_inst,
                                           IRBuilder* builder) {
  switch (tf_inst->GetExtOpCode()) {
    case TFExtOpCode::ADDN: {
      return ConvertAddN(tf_inst, builder);
    }
    case TFExtOpCode::BROADCASTTO: {
      return ConvertBroadcastTo(tf_inst, builder);
    }
    case TFExtOpCode::CAST: {
      return ConvertCast(tf_inst, builder);
    }
    case TFExtOpCode::EXPANDDIMS: {
      return ConvertExpandDims(tf_inst, builder);
    }
    case TFExtOpCode::FILL: {
      return ConvertFill(tf_inst, builder);
    }
    case TFExtOpCode::MERGE: {
      return ConvertMerge(tf_inst, builder);
    }
    case TFExtOpCode::STOPGRADIENT:
    case TFExtOpCode::QUEUEDEQUEUEV2:
    case TFExtOpCode::IDENTITY: {
      return {tf_inst->GetOperand(0)};
    }
    case TFExtOpCode::SIZE: {
      return ConvertSize(tf_inst, builder);
    }
    case TFExtOpCode::SPLIT: {
      return ConvertSplit(tf_inst, builder);
    }
    case TFExtOpCode::SQUARE: {
      return ConvertSquare(tf_inst, builder);
    }
    case TFExtOpCode::SQUEEZE: {
      return ConvertSqueeze(tf_inst, builder);
    }
    case TFExtOpCode::STRIDEDSLICE: {
      return ConvertStridedSlice(tf_inst, builder);
    }
    case TFExtOpCode::SWITCH: {
      return ConvertSwitch(tf_inst, builder);
    }
    case TFExtOpCode::RESHAPE: {
      return ConvertReshape(tf_inst, builder);
    }
    case TFExtOpCode::BATCHTOSPACEND: {
      return ConvertBatch2Space(tf_inst, builder);
    }
    case TFExtOpCode::SPACETOBATCHND: {
      return ConvertSpace2Batch(tf_inst, builder);
    }
    case TFExtOpCode::FIFOQUEUEV2: {
      return GetPlaceholder(tf_inst, builder);
    }
    // Todo: DNNl and ODLA should support quant/dequant op, for performace
    // considerations; Now split the Hgai quant op to
    // Mul+Add+Round+Cast+Max+Min+Transpose; Split Hgai dequant op to
    // Cast+Mul+Add+Tranpose;
    case TFExtOpCode::HGQUANT: {
      return ConvertHgQuant(tf_inst, builder);
    }
    case TFExtOpCode::HGDEQUANT: {
      return ConvertHgDeQuant(tf_inst, builder);
    }
    case TFExtOpCode::ZEROSLIKE: {
      return ConvertZerosLike(tf_inst, builder);
    }
    default: {
      tf_inst->Dump();
      HLCHECK(0 && "Unhandled");
    }
  }
  return std::vector<Def>{};
}

bool TFExtensionLegalizer::RunOnBasicBlock(BasicBlock* bb) {
  IRBuilder builder(bb);
  bool changed = false;
  // Insert ouput before extension legalization.
  changed |= AppendReturnInst(bb);

  for (auto& inst_t : *bb) {
    Instruction* inst = inst_t.get();
    if (inst->GetOpCode() == OpCode::EXTENSION) {
      ExtensionInst* ext_inst = Downcast<ExtensionInst>(inst);
      if (ext_inst->GetExtensionKind() ==
          ExtensionInst::ExtensionKind::kExtension_TENSORFLOW) {
        TFExtensionInst* tf_inst = Downcast<TFExtensionInst>(ext_inst);
        std::vector<Def> new_defs;
        if (tf_inst->GetExtOpCode() == TFExtOpCode::SPLIT &&
            !convert_split_to_slice_) {
          new_defs = ConvertTFSplit(tf_inst, &builder);
        } else {
          new_defs = ConvertTFExtension(tf_inst, &builder);
        }
        if (!new_defs.empty()) {
          tf_inst->ReplaceAllUsesWith(new_defs);
        }
      }
    } else if (inst->GetOpCode() == OpCode::CONV2D) {
      Conv2DInst* conv_inst = Downcast<Conv2DInst>(inst);
      const auto& op1_type = conv_inst->GetOperand(1).GetType();
      if (conv_inst->GetGroup() == 0 && op1_type.IsValid()) {
        auto op1_dims = op1_type.GetDimSizes();
        HLCHECK(op1_dims.size() >= 4);
        conv_inst->SetGroup(static_cast<int>(op1_dims[op1_dims.size() - 2]));
        // Normalize HWIM to HWIO
        auto p = op1_dims[2] * op1_dims[3];
        op1_dims[3] *= conv_inst->GetGroup();
        HLCHECK(p % op1_dims[3] == 0);
        op1_dims[2] = p / op1_dims[3];
        conv_inst->GetOperand(1).SetType(
            halo::Type{op1_type.GetDataType(), op1_dims});
      }
    } else if (inst->GetOpCode() == OpCode::ONEHOT) {
      OneHotInst* one_hot = Downcast<OneHotInst>(inst);
      changed |= FixUpOneHot(one_hot, &builder);
    }
  }
  return changed;
}

} // end namespace halo
