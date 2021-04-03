//===- inst_simplify.cc ---------------------------------------------------===//
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

#include "halo/lib/transforms/inst_simplify.h"

#include <algorithm>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <unordered_set>

#include "halo/api/halo_data.h"
#include "halo/lib/framework/common.h"
#include "halo/lib/framework/data_layout.h"
#include "halo/lib/framework/global_context.h"
#include "halo/lib/ir/common_cast_instructions.h"
#include "halo/lib/ir/common_instructions.h"
#include "halo/lib/ir/common_reduction_instructions.h"
#include "halo/lib/ir/ir_builder.h"
#include "halo/lib/ir/math_instructions.h"
#include "halo/lib/ir/nn_cnn_instructions.h"
#include "halo/lib/transforms/transforms_util.h"
#include "halo/lib/transforms/type_legalizer.h"

namespace halo {

template <typename T>
static Constant* RunConstantFoldingOnMathBinary(const std::string& name,
                                                const Type& ret_type, Def op0,
                                                Def op1, OpCode opcode,
                                                KindPredicate pred) {
  if (!IsA<Constant>(op0) || !IsA<Constant>(op1) ||
      op0.GetType().GetTotalNumOfElements() !=
          op1.GetType().GetTotalNumOfElements()) {
    return nullptr;
  }
  if (opcode == OpCode::CMP) {
    if (pred == KindPredicate::GE) {
      pred = KindPredicate::LT;
      std::swap(op0, op1);
    } else if (pred == KindPredicate::GT) {
      pred = KindPredicate::LE;
      std::swap(op0, op1);
    }
  }
  Constant* c_lhs = DynCast<Constant>(op0.GetOwner());
  Constant* c_rhs = DynCast<Constant>(op1.GetOwner());
  size_t num_elements = op0.GetType().GetTotalNumOfElements();
  Constant* c_ret = nullptr;
  ConstantBuilder cb(DynCast<Function>(c_lhs->GetParent()));
  std::vector<T> ret;
  ret.reserve(num_elements);

  switch (opcode) {
    case OpCode::ADD: {
      for (size_t i = 0; i < num_elements; ++i) {
        ret.push_back(c_lhs->GetData<T>(i) + c_rhs->GetData<T>(i));
      }
      c_ret = cb.CreateConstant(name, ret_type, ret.data());
      break;
    }
    case OpCode::MUL: {
      for (size_t i = 0; i < num_elements; ++i) {
        ret.push_back(c_lhs->GetData<T>(i) * c_rhs->GetData<T>(i));
      }
      c_ret = cb.CreateConstant(name, ret_type, ret.data());
      break;
    }
    case OpCode::DIV: {
      for (size_t i = 0; i < num_elements; ++i) {
        ret.push_back(c_lhs->GetData<T>(i) / c_rhs->GetData<T>(i));
      }
      c_ret = cb.CreateConstant(name, ret_type, ret.data());
      break;
    }
    case OpCode::CMP: {
      std::vector<int8_t> ret;
      switch (pred) {
        case KindPredicate::LT: {
          for (size_t i = 0; i < num_elements; ++i) {
            if (c_lhs->GetData<T>(i) < c_rhs->GetData<T>(i)) {
              ret.push_back(1);
            } else {
              ret.push_back(0);
            }
          }
          c_ret = cb.CreateConstant(name, ret_type, ret.data());
          break;
        }
        default: {
          break;
        }
      }
      break;
    }
    default: {
      break;
    }
  }
  return c_ret;
}

template <typename T>
static T* FuseToConvDeConv(const T* conv, OpCode opc, const Constant* c) {
  HLCHECK(IsA<Constant>(conv->GetOperand(1)));
  const auto kernel = DynCast<Constant>(conv->GetOperand(1));
  const auto& kernel_type = kernel->GetResultType();
  const auto& info = ImageAxisInfo::GetImageAxisInfo(conv->GetDataFormat(),
                                                     conv->GetFilterFormat());
  auto group = conv->GetGroup();
  if (group < 1 || !conv->GetResultType().IsValid()) {
    return nullptr;
  }
  unsigned output_dim = info.kernel_output_axis;

  auto output_ch =
      conv->GetResultType().GetNumOfElementsInDim(info.data_channel_axis);

  bool has_valid_bias =
      conv->GetNumOfOperands() == 2 ||
      (conv->GetNumOfOperands() == 3 && IsA<Constant>(conv->GetOperand(2)) &&
       conv->GetOperand(2).GetType().GetTotalNumOfElements() == output_ch);
  if (!has_valid_bias) {
    return nullptr;
  }
  ConstantBuilder cb(conv->GetParent()->GetParent());
  IRBuilder builder(conv->GetParent());
  builder.SetInsertAfter(conv);

  auto n = c->GetResultType().GetTotalNumOfElements();
  if (!has_valid_bias || output_ch != n) {
    return nullptr;
  }

  auto ops = conv->GetOperands();
  const Constant* op_bias = conv->GetNumOfOperands() == 3
                                ? DynCast<Constant>(conv->GetOperand(2))
                                : nullptr;
  if (opc == OpCode::MUL) {
    std::vector<float> data(kernel_type.GetTotalNumOfElements());
    size_t extend = 1;
    for (auto i = kernel_type.GetNumOfDims() - 1; i > output_dim; --i) {
      extend *= kernel_type.GetNumOfElementsInDim(i);
    }
    for (size_t i = 0, e = data.size(); i < e; ++i) {
      auto idx = (i / extend) % n;
      data[i] = kernel->template GetData<float>(i) * c->GetData<float>(idx);
    }
    auto new_kernel =
        cb.CreateConstant(kernel->GetName(), kernel_type, data.data());
    ops[1] = *new_kernel;
    if (op_bias != nullptr) {
      std::vector<float> data(output_ch);
      for (int i = 0; i < output_ch; ++i) {
        data[i] = op_bias->GetData<float>(i) * c->GetData<float>(i);
      }
      auto new_bias = cb.CreateConstant(op_bias->GetName(),
                                        op_bias->GetResultType(), data.data());
      ops[2] = *new_bias;
    }
  } else {
    std::vector<int64_t> shape(kernel_type.GetNumOfDims(), 1);
    shape[info.data_channel_axis] = n;
    halo::Type ty{kernel_type.GetDataType(), shape};
    std::vector<float> data(output_ch);
    for (int i = 0; i < output_ch; ++i) {
      data[i] = (op_bias == nullptr ? 0 : op_bias->GetData<float>(i)) +
                c->GetData<float>(i);
    }
    auto new_bias = cb.CreateConstant(c->GetName(), ty, data.data());
    if (ops.size() == 3) {
      ops.pop_back();
    }
    ops.push_back(*new_bias);
  }
  auto new_conv = builder.Clone(*conv, ops);
  return DynCast<T>(new_conv);
}

static std::pair<Def, Def> RunOnMathBinaryInstruction(Instruction* binary_inst,
                                                      bool disable_broadcasting,
                                                      bool fuse_conv_bias) {
  Def orig_def{binary_inst, 0};
  auto op0 = binary_inst->GetOperand(0);
  auto op1 = binary_inst->GetOperand(1);
  auto opc = binary_inst->GetOpCode();
  bool has_swapped = false;
  if (IsA<Constant>(op0)) {
    std::swap(op0, op1);
    has_swapped = true;
  }

  // MUL(x, 1) ==> x.
  if (opc == OpCode::MUL && IsA<Constant>(op1)) {
    const Constant* c = DynCast<Constant>(op1);
    if (c->HasSameValueOf(1)) {
      return {orig_def, op0};
    }
  }

  // ADD/SUB(x, 0) ==> x.
  if ((opc == OpCode::ADD || opc == OpCode::SUB) && IsA<Constant>(op1)) {
    const Constant* c = DynCast<Constant>(op1);
    if (c->HasSameValueOf(0)) {
      return {orig_def, op0};
    }
  }

  IRBuilder builder(binary_inst->GetParent());
  builder.SetInsertAfter(binary_inst);
  ConstantBuilder cb(binary_inst->GetParent()->GetParent());

  // Fuse mul/add into conv.
  if ((opc == OpCode::MUL || (fuse_conv_bias && opc == OpCode::ADD)) &&
      IsA<Constant>(op1)) {
    const Constant* c = DynCast<Constant>(op1);
    // check if mul can be fused with conv
    Instruction* new_inst = nullptr;
    if (IsA<Conv2DInst>(op0)) {
      new_inst = FuseToConvDeConv(DynCast<Conv2DInst>(op0), opc, c);
    } else if (IsA<Conv2DTransposeInst>(op0)) {
      new_inst = FuseToConvDeConv(DynCast<Conv2DTransposeInst>(op0), opc, c);
    }
    if (new_inst != nullptr) {
      return {orig_def, *new_inst};
    }
  }

  const auto& op0_type = op0.GetType();
  const auto& op1_type = op1.GetType();
  OpCode opcode = binary_inst->GetOpCode();
  /*
  // Handle scalar constant
  if (IsA<Constant>(op1.GetOwner())) {
    Constant* c_op1 = DynCast<Constant>(op1.GetOwner());
    Type ret_type = binary_inst->GetResultsTypes()[0];
    HLCHECK(ret_type.IsValid());
    if (c_op1->IsScalarZero()) {
      if (opcode == OpCode::ADD) {
        return {orig_def, op0};
      }
      if (opcode == OpCode::MUL) {
        Constant* c_zero =
            cb.SplatConstantZero(binary_inst->GetName(), ret_type);
        return {orig_def, *c_zero};
      }
    }
    if (c_op1->IsScalarOne()) {
      if (opcode == OpCode::MUL) {
        return {orig_def, op0};
      }
    }
  }*/

  const int64_t folding_threshold = 10;
  // Both operands are constant, do constant folding
  if (IsA<Constant>(op0) && IsA<Constant>(op1) &&
      op0_type.GetTotalNumOfElements() == op1_type.GetTotalNumOfElements() &&
      op0_type.GetTotalNumOfElements() < folding_threshold) {
    Type ret_type = binary_inst->GetResultsTypes()[0];
    HLCHECK(ret_type.IsValid());
    KindPredicate pred = KindPredicate::INVALID;
    if (opcode == OpCode::CMP) {
      pred = static_cast<CmpInst*>(binary_inst)->GetPredicator(); // NOLINT
    }
    if (has_swapped) {
      std::swap(op0, op1);
    }
    Constant* c_ret = nullptr;
    switch (op0_type.GetDataType()) {
      case DataType::INT32: {
        c_ret = RunConstantFoldingOnMathBinary<int>(
            binary_inst->GetName() + "_folding", ret_type, op0, op1, opcode,
            pred);
        break;
      }
      case DataType::INT64: {
        c_ret = RunConstantFoldingOnMathBinary<int64_t>(
            binary_inst->GetName() + "_folding", ret_type, op0, op1, opcode,
            pred);
        break;
      }
      case DataType::FLOAT32: {
        c_ret = RunConstantFoldingOnMathBinary<float>(
            binary_inst->GetName() + "_folding", ret_type, op0, op1, opcode,
            pred);
        break;
      }
      default:
        c_ret = nullptr;
    }
    if (c_ret != nullptr) {
      return {orig_def, *c_ret};
    }
    return {orig_def, orig_def};
  }

  // Do offline broadcasting.
  if (!disable_broadcasting && op0_type.IsValid() && IsA<Constant>(op1) &&
      op0_type.GetNumOfDims() != op1_type.GetNumOfDims() &&
      op0_type.GetTotalNumOfElements() >= op1_type.GetTotalNumOfElements() &&
      op0_type.GetNumOfElementsInDim(op0_type.GetNumOfDims() - 1) != 1) {
    size_t lhs_cnt = op0_type.GetTotalNumOfElements();
    size_t rhs_cnt = op1_type.GetTotalNumOfElements();
    auto orig_addend = DynCast<Constant>(op1.GetOwner());
    HLCHECK(lhs_cnt % rhs_cnt == 0);
    HLCHECK(op1_type.GetDataType() == op0_type.GetDataType());

    auto copies = lhs_cnt / rhs_cnt;

    size_t copy_size = rhs_cnt * orig_addend->GetElementSizeInBytes();
    std::vector<char> buf(copies * copy_size);
    for (size_t i = 0; i < copies; ++i) {
      memcpy(&buf[copy_size * i], orig_addend->GetRawDataPtr(), copy_size);
    }
    auto addend = cb.CreateConstant(orig_addend->GetName() + "_broadcasted_" +
                                        std::to_string(binary_inst->GetId()),
                                    op0_type, buf.data());
    auto new_add = has_swapped ? builder.CreateBinary(binary_inst->GetName(),
                                                      *addend, op0, opcode)
                               : builder.CreateBinary(binary_inst->GetName(),
                                                      op0, *addend, opcode);
    new_add->GetResultsTypes()[0] = binary_inst->GetResultsTypes()[0];
    return {orig_def, *new_add};
  }

  if (op0_type.IsValid() && IsA<Constant>(op1) && IsA<TransposeInst>(op0) &&
      op1_type.BroadcastableTo(op0_type)) {
    Instruction* op0_inst = DynCast<Instruction>(op0);
    if (!IsA<Argument>(op0_inst->GetOperand(0))) {
      // Add(transpose(op0), op1) ==> transpose(add(op0, transpose'(op1))
      TransposeInst* orig_transpose = DynCast<TransposeInst>(op0_inst);
      IRBuilder builder(binary_inst->GetParent());
      builder.SetInsertAfter(binary_inst);
      const auto& orig_perm = orig_transpose->GetPermutation();
      Instruction* new_op1 = nullptr;
      if (op1_type.GetSqueezedNumOfDims() == 1) {
        auto dims = std::vector<int64_t>(op0_type.GetNumOfDims(), 1);
        int64_t op1_vector_axis = op0_type.GetNumOfDims() - 1;
        for (auto n = op1_type.GetTotalNumOfElements(); op1_vector_axis >= 0;
             --op1_vector_axis) {
          if (op0_type.GetNumOfElementsInDim(op1_vector_axis) == n) {
            break;
          }
        }
        dims[orig_perm[op1_vector_axis]] = op1_type.GetTotalNumOfElements();
        ConstantBuilder cb(binary_inst->GetParent()->GetParent());
        Constant* c_shape = cb.CreateConstant(
            op1.GetDef()->GetName() + "_shape",
            halo::Type{DataType::INT64, {static_cast<int64_t>(dims.size())}},
            dims.data());
        auto new_addend = builder.CreateReshape(op1.GetDef()->GetName() + "_r",
                                                {op1, *c_shape});
        new_op1 = new_addend;
        new_addend->GetResultsTypes()[0] =
            Type{op1.GetType().GetDataType(), dims};
      } else {
        auto reverse_perm = orig_perm;
        for (int i = 0, e = orig_perm.size(); i < e; ++i) {
          reverse_perm[orig_perm[i]] = i;
        }
        auto new_addend =
            builder.CreateTranspose(op1.GetDef()->GetName() + "_t", {op1});
        new_addend->SetPermutation(reverse_perm);
        new_op1 = new_addend;
      }

      auto new_binary =
          builder.CreateBinary(binary_inst->GetName(),
                               op0.GetDef()->GetOperand(0), *new_op1, opcode);
      TransposeInst* new_transpose =
          builder.CreateTranspose("t_" + binary_inst->GetName(), {*new_binary});
      new_transpose->SetPermutation(orig_perm);
      return {orig_def, *new_transpose};
    }
  }

  // add(transpose(v0), transpose(v1)) => transpose(v0, v1)
  if (IsA<TransposeInst>(op0) && IsA<TransposeInst>(op1)) {
    const TransposeInst* tr0 = DynCast<const TransposeInst>(op0);
    const TransposeInst* tr1 = DynCast<const TransposeInst>(op1);
    if (tr0->GetPermutation() == tr1->GetPermutation()) {
      IRBuilder builder(binary_inst->GetParent());
      builder.SetInsertAfter(binary_inst);
      auto new_binary = builder.CreateAdd(
          binary_inst->GetName(), tr0->GetOperand(0), tr1->GetOperand(0));
      TransposeInst* new_tr =
          builder.CreateTranspose(binary_inst->GetName() + "_t", {*new_binary});
      new_tr->SetPermutation(tr0->GetPermutation());
      return {orig_def, *new_tr};
    }
  }

  return {orig_def, orig_def};
}

template <typename ReduceInstTy, typename Build>
static std::pair<Def, Def> EliminateTranspose(ReduceInstTy* inst, Build build) {
  Def orig_def{inst, 0};
  std::pair<Def, Def> ret{orig_def, orig_def};

  // ReduceMean(tranpose(x, {t0, t1, t2, t3}, {a0, a1, a2...}) => ReduceMean(x,
  // permed_axis)
  Def op0 = inst->GetOperand(0);
  if (IsA<TransposeInst>(op0)) {
    IRBuilder builder(inst->GetParent());
    builder.SetInsertAfter(inst);
    const TransposeInst* transpose = DynCast<TransposeInst>(op0);
    const auto& perm = transpose->GetPermutation();
    const auto& orig_axes = inst->GetAxis();
    std::vector<int> new_axes(orig_axes.size());
    std::transform(orig_axes.begin(), orig_axes.end(), new_axes.begin(),
                   [&perm](int x) { return perm[x]; });
    ReduceInstTy* new_inst =
        build(builder, inst->GetName(), transpose->GetOperand(0));
    new_inst->SetAxis(new_axes);
    ret.second = *new_inst;
    return ret;
  }
  return ret;
}

static std::pair<Def, Def> RunOnCommonReductionInstruction(Instruction* inst) {
  Def orig_def{inst, 0};

  auto op0 = inst->GetOperand(0);
  const Type& dst_type = inst->GetResultsTypes()[0];
  const Type& op0_type = op0.GetType();
  OpCode opcode = inst->GetOpCode();

  // Move the constant axis into attribute.
  if (inst->GetNumOfOperands() > 1 && IsA<Constant>(inst->GetOperand(1))) {
    const Constant* data = DynCast<Constant>(inst->GetOperand(1).GetOwner());
    const Type& ty = data->GetResultType();
    if (ty.GetDataType() == DataType::INT32) {
      const int32_t* ptr = data->GetDataPtr<int32_t>();
      std::vector<int> axis(ptr, ptr + ty.GetTotalNumOfElements()); // NOLINT
      IRBuilder builder(inst->GetParent());
      builder.SetInsertAfter(inst);
      Instruction* ret = nullptr;
      switch (opcode) {
        case OpCode::REDUCEMEAN: {
          ReduceMeanInst* new_inst = DynCast<ReduceMeanInst>(
              builder.Clone(*inst, {inst->GetOperand(0)}));
          new_inst->SetAxis(axis);
          ret = new_inst;
          break;
        }
        case OpCode::ARGMAX: {
          ArgmaxInst* new_inst =
              DynCast<ArgmaxInst>(builder.Clone(*inst, {inst->GetOperand(0)}));
          new_inst->SetAxis(axis.at(0));
          ret = new_inst;
          break;
        }
        default: {
          break;
        }
      }
      if (ret != nullptr) {
        ret->GetResultsTypes()[0] = inst->GetResultType();
        return {orig_def, *ret};
      }
    }
  }

  if (inst->GetNumOfOperands() == 1) {
    std::pair<Def, Def> ret{orig_def, orig_def};
    switch (opcode) {
      case OpCode::REDUCEMIN: {
        ret = EliminateTranspose(
            DynCast<ReduceMinInst>(inst),
            [](IRBuilder& builder, const std::string& name, const Def& def) {
              return builder.CreateReduceMin(name, {def});
            });
        break;
      }
      case OpCode::REDUCEMAX: {
        ret = EliminateTranspose(
            DynCast<ReduceMaxInst>(inst),
            [](IRBuilder& builder, const std::string& name, const Def& def) {
              return builder.CreateReduceMax(name, {def});
            });
        break;
      }
      case OpCode::REDUCEMEAN: {
        ret = EliminateTranspose(
            DynCast<ReduceMeanInst>(inst),
            [](IRBuilder& builder, const std::string& name, const Def& def) {
              return builder.CreateReduceMean(name, {def});
            });
        break;
      }
      case OpCode::REDUCEPRODUCT: {
        ret = EliminateTranspose(
            DynCast<ReduceProductInst>(inst),
            [](IRBuilder& builder, const std::string& name, const Def& def) {
              return builder.CreateReduceProduct(name, {def});
            });
        break;
      }
      default: {
        break;
      }
    }
    if (ret.first != ret.second) {
      return ret;
    }
  }

  if (!dst_type.IsValid() || !IsA<Constant>(op0.GetOwner()) ||
      op0_type.GetNumOfDims() > 1) {
    return {orig_def, orig_def};
  }

  ConstantBuilder cb(inst->GetParent()->GetParent());
  Constant* c_input = DynCast<Constant>(op0.GetOwner());
  DataType dt = op0_type.GetDataType();

  if (op0_type.GetTotalNumOfElements() == 1) {
    Constant* c_ret = nullptr;
    if (opcode == OpCode::ARGMAX || opcode == OpCode::ARGMIN) {
      int ret = 0;
      c_ret = cb.CreateConstant(inst->GetName() + "_folding", dst_type, &ret);
    } else {
      c_ret = cb.CreateConstant(inst->GetName() + "_folding", dst_type,
                                c_input->GetRawDataPtr());
    }
    return {orig_def, *c_ret};
  }
  if (dt == DataType::INT32) {
    switch (opcode) {
      case OpCode::REDUCEMIN:
      case OpCode::REDUCEMAX:
      case OpCode::ARGMAX: {
        int ret = std::numeric_limits<int32_t>::lowest();
        int index = -1;
        for (int i = 0; i < op0_type.GetTotalNumOfElements(); ++i) {
          int data_i = c_input->GetData<int>(i);
          ret = std::max(ret, data_i);
          if (ret == data_i) {
            index = i;
          }
        }
        if (opcode == OpCode::REDUCEMAX || opcode == OpCode::REDUCEMIN) {
          auto new_def =
              cb.CreateConstant(inst->GetName() + "_folding", dst_type, &ret);
          return {orig_def, *new_def};
        }
        // ARGMAX
        auto new_def =
            cb.CreateConstant(inst->GetName() + "_folding", dst_type, &index);
        return {orig_def, *new_def};
      }
      case OpCode::REDUCEMEAN:
      case OpCode::REDUCESUM: {
        int ret = 0;
        for (int i = 0; i < op0_type.GetTotalNumOfElements(); ++i) {
          ret += c_input->GetData<int>(i);
        }
        if (opcode == OpCode::REDUCEMEAN) {
          ret /= op0_type.GetTotalNumOfElements();
        }
        auto new_def =
            cb.CreateConstant(inst->GetName() + "_folding", dst_type, &ret);
        return {orig_def, *new_def};
      }
      case OpCode::REDUCEPRODUCT: {
        int ret = 1;
        for (int i = 0; i < op0_type.GetTotalNumOfElements(); ++i) {
          ret *= c_input->GetData<int>(i);
        }
        auto new_def =
            cb.CreateConstant(inst->GetName() + "_folding", dst_type, &ret);
        return {orig_def, *new_def};
      }
      default: {
        return {orig_def, orig_def};
      }
    }
  }
  return {orig_def, orig_def};
}

/// By default, nothing is updated.
std::pair<Def, Def> InstSimplify::RunOnInstruction(Instruction* inst) {
  switch (inst->GetOpCode()) {
    case OpCode::ADD:
    case OpCode::MUL:
    case OpCode::DIV:
    case OpCode::SUB:
    case OpCode::CMP: {
      return RunOnMathBinaryInstruction(inst, disable_broadcasting_,
                                        fuse_conv_bias_);
    }
    case OpCode::REDUCEMAX:
    case OpCode::REDUCEMIN:
    case OpCode::REDUCEMEAN:
    case OpCode::REDUCESUM:
    case OpCode::REDUCEPRODUCT:
    case OpCode::ARGMAX:
    case OpCode::ARGMIN: {
      return RunOnCommonReductionInstruction(inst);
    }
    default: {
      return std::make_pair(Def{inst, 0}, Def{inst, 0});
    }
  }
}

template <typename InstType, typename Builder>
static std::pair<Def, Def> SinkTranspose(InstType& inst, Builder build) {
  std::pair<Def, Def> ret{Def{&inst, 0}, Def{&inst, 0}};

  if (const auto& op0 = inst.GetOperand(0); IsA<TransposeInst>(op0)) {
    // Inst(transpose(x)) -> transpose(Inst(x)), this exposes opportunites
    // to cancel out transposes.
    const TransposeInst* orig_trans = DynCast<TransposeInst>(op0);
    IRBuilder builder(inst.GetParent());
    builder.SetInsertAfter(&inst);
    InstType* new_inst =
        build(builder, inst.GetName(), op0.GetDef()->GetOperand(0));
    TransposeInst* new_trans =
        builder.CreateTranspose(inst.GetName() + "_t", {*new_inst});
    new_trans->SetPermutation(orig_trans->GetPermutation());
    ret.second = *new_trans;
    return ret;
  }
  return ret;
}

std::pair<Def, Def> InstSimplify::RunOnInstruction(LeakyReluInst* inst) {
  return SinkTranspose(*inst, [inst](IRBuilder& builder,
                                     const std::string& name, const Def& op) {
    auto new_inst = builder.CreateLeakyRelu(name, op);
    new_inst->SetAlpha(inst->GetAlpha());
    return new_inst;
  });
}

std::pair<Def, Def> InstSimplify::RunOnInstruction(PReluInst* inst) {
  auto op1 = inst->GetOperand(1);
  return SinkTranspose(
      *inst,
      [inst, &op1](IRBuilder& builder, const std::string& name, const Def& op) {
        return DynCast<PReluInst>(builder.Clone(*inst, {op, op1}));
      });
}

template <typename T>
static Constant* GetPermutedConstant(ConstantBuilder* cb, const Constant* orig,
                                     const std::vector<int32_t>& perm) {
  const auto& shape_type = orig->GetResultType();
  auto ranks = shape_type.GetTotalNumOfElements();
  std::vector<T> data(ranks);
  for (int64_t i = 0; i < ranks; ++i) {
    data[perm[i]] = orig->GetData<T>(i);
  }
  return cb->CreateConstant(orig->GetName(), shape_type, data.data());
}

static bool IsNullConstant(const Def& op) {
  if (IsA<Constant>(op)) {
    auto type = DynCast<Constant>(op)->GetResultType();
    if (!type.IsScalar() && type.GetTotalNumOfElements() == 0) {
      return true;
    }
  }
  return false;
}

std::pair<Def, Def> InstSimplify::RunOnInstruction(ResizeInst* inst) {
  Def orig_def{inst, 0};
  // Check if the optional operand is valid or not.
  // A null constant can be ignored.
  std::vector<Def> valid_operands;
  for (const auto& op : inst->GetOperands()) {
    if (!IsNullConstant(op)) {
      valid_operands.push_back(op);
    }
  }
  if (valid_operands.size() < inst->GetNumOfOperands()) {
    IRBuilder builder(inst->GetParent());
    builder.SetInsertAfter(inst);
    // Remove invalid operands.
    auto new_resize = builder.Clone(*inst, valid_operands);
    return {orig_def, *new_resize};
  }
  // Resize with 3 operands are not handled.
  HLCHECK(inst->GetNumOfOperands() <= 2);
  if (const auto& op0 = inst->GetOperand(0); IsA<TransposeInst>(op0)) {
    if (auto op1 = inst->GetOperand(1); IsA<Constant>(op1)) {
      Constant* shape = DynCast<Constant>(op1);
      const auto& shape_type = shape->GetResultType();
      ConstantBuilder cb(inst->GetParent()->GetParent());
      auto orig_perm = DynCast<TransposeInst>(op0)->GetPermutation();
      Constant* new_shape = nullptr;
      switch (shape_type.GetDataType()) {
        case DataType::INT32: {
          new_shape = GetPermutedConstant<int32_t>(&cb, shape, orig_perm);
          break;
        }
        case DataType::INT64: {
          new_shape = GetPermutedConstant<int64_t>(&cb, shape, orig_perm);
          break;
        }
        case DataType::FLOAT32: {
          new_shape = GetPermutedConstant<float>(&cb, shape, orig_perm);
          break;
        }
        default:
          HLCHECK(0 && "Invalid resize shape type");
      }
      new_shape->SetName(inst->GetName() + "_resize_shape");

      return SinkTranspose(
          *inst, [new_shape, inst](IRBuilder& builder, const std::string& name,
                                   const Def& op) {
            auto new_inst = builder.CreateResize(name, {op, *new_shape});
            new_inst->CopyAttrsFrom(*inst);
            return new_inst;
          });
    }
  }
  return {orig_def, orig_def};
}

std::pair<Def, Def> InstSimplify::RunOnInstruction(Relu6Inst* inst) {
  return SinkTranspose(
      *inst, [](IRBuilder& builder, const std::string& name, const Def& op) {
        return builder.CreateRelu6(name, op);
      });
}

std::pair<Def, Def> InstSimplify::RunOnInstruction(SigmoidInst* inst) {
  return SinkTranspose(
      *inst, [](IRBuilder& builder, const std::string& name, const Def& op) {
        return builder.CreateSigmoid(name, op);
      });
}

std::pair<Def, Def> InstSimplify::RunOnInstruction(ReluInst* inst) {
  return SinkTranspose(
      *inst, [](IRBuilder& builder, const std::string& name, const Def& op) {
        return builder.CreateRelu(name, op);
      });
}

std::pair<Def, Def> InstSimplify::RunOnInstruction(Conv2DInst* inst) {
  std::pair<Def, Def> ret{Def{inst, 0}, Def{inst, 0}};
  if (!inst->GetResultType().IsValid()) {
    return ret;
  }

  Def op_input = inst->GetOperand(0);
  Def op_kernel = inst->GetOperand(1);

  IRBuilder builder(inst->GetParent());
  builder.SetInsertAfter(inst);

  // Convert conv(pad(x, amt), kernel) to conv(x, kernel) to eliminate
  // pad op.
  if (IsA<PadInst>(op_input)) {
    const PadInst* pad = DynCast<PadInst>(op_input);
    Def pad_op0 = pad->GetOperand(0);
    Def pad_op1 = pad->GetOperand(1);
    if (IsA<Constant>(pad_op1.GetOwner())) {
      const Constant* pad_amt = DynCast<Constant>(pad_op1);
      unsigned dims = pad_amt->GetResultType().GetNumOfDims();
      if (dims == 2 &&
          pad_amt->GetResultType().GetTotalNumOfElements() == 4 * dims) {
        std::vector<int32_t> vals(
            pad_amt->GetDataPtr<int32_t>(),
            pad_amt->GetDataPtr<int32_t>() + 4 * dims); // NOLINT
        if (vals[0] != 0 || vals[1] != 0) {
          return ret;
        }

        const auto& info = ImageAxisInfo::GetImageAxisInfo(
            inst->GetDataFormat(), inst->GetFilterFormat());

        std::array<int, 4> indices_nc = {0, 1, info.data_channel_axis * 2,
                                         info.data_channel_axis * 2 + 1};
        // No paddings on N & C.
        for (auto idx : indices_nc) {
          if (vals[idx] != 0) {
            return ret;
          }
        }
        std::array<int, 4> indices_hw = {
            info.data_height_axis * 2, info.data_height_axis * 2 + 1,
            info.data_width_axis * 2, info.data_width_axis + 1};

        Conv2DInst* new_inst =
            builder.CreateConv2D(inst->GetName(), {pad_op0, op_kernel});
        new_inst->SetDataFormat(inst->GetDataFormat());
        new_inst->SetFilterFormat(inst->GetFilterFormat());
        new_inst->SetDilations(inst->GetDilations());
        new_inst->SetStrides(inst->GetStrides());
        new_inst->SetPaddingTop(inst->GetPaddingTop() + vals[indices_hw[0]]);
        new_inst->SetPaddingBottom(inst->GetPaddingBottom() +
                                   vals[indices_hw[1]]);
        new_inst->SetPaddingLeft(inst->GetPaddingLeft() + vals[indices_hw[2]]);
        new_inst->SetPaddingRight(inst->GetPaddingRight() +
                                  vals[indices_hw[3]]);
        new_inst->GetResultsTypes()[0] = inst->GetResultsTypes()[0];
        new_inst->SetPadding(Padding::EXPLICIT);
        ret.second = Def(new_inst, 0);
      }
    }
  }

  // Convert Conv(add(x, c), k) => Conv(x, k') or  Conv(mul(x, c), k) ==>
  // Conv(x, k') where k is a constant of scalar or channel-wise vector.
  if ((IsA<AddInst>(op_input) || IsA<MulInst>(op_input)) &&
      IsA<Constant>(op_kernel) && inst->GetGroup() == 1 &&
      inst->GetResultType().IsValid()) {
    Instruction* binary_inst = DynCast<Instruction>(op_input);
    auto binary_op0 = binary_inst->GetOperand(0);
    if (IsA<Conv2DInst>(binary_op0)) {
      // For pattens like a = conv(); b = a + c; d = conv(b), prefer to fuse a
      // and b.
      return ret;
    }

    auto binary_op1 = binary_inst->GetOperand(1);
    if (!IsA<Constant>(binary_op1)) {
      return ret;
    }
    const auto& kernel_type = op_kernel.GetType();
    Constant* c = DynCast<Constant>(binary_op1);
    if (kernel_type.GetDataType() != DataType::FLOAT32 ||
        kernel_type.GetDataType() != c->GetResultType().GetDataType()) {
      return ret;
    }

    // match shape of C: expect [..,in_chs, 1, 1].
    auto n_elems = c->GetResultType().GetTotalNumOfElements();
    auto dims = c->GetResultType().GetDimSizes();
    auto kernel_shape = kernel_type.GetDimSizes();
    const auto& info = ImageAxisInfo::GetImageAxisInfo(inst->GetDataFormat(),
                                                       inst->GetFilterFormat());
    auto in_chs = kernel_shape[info.kernel_input_axis];

    auto in_chs_dim_r =
        info.kernel_input_axis - kernel_shape.size(); // Dims in backwards.
    if (!(n_elems == in_chs &&
          (dims.size() == 1 || (-in_chs_dim_r <= dims.size() &&
                                dims[dims.size() + in_chs_dim_r] == in_chs)))) {
      return ret;
    }

    bool has_padding =
        inst->GetPaddingBottom() != 0 || inst->GetPaddingLeft() != 0 ||
        inst->GetPaddingTop() != 0 || inst->GetPaddingRight() != 0;

    Constant* kernel = DynCast<Constant>(op_kernel);
    ConstantBuilder cb(inst->GetParent()->GetParent());

    std::vector<float> new_kernel_data(kernel_type.GetTotalNumOfElements());
    auto operands = inst->GetOperands();
    operands[0] = binary_op0;

    std::vector<size_t> strides(kernel_shape.size(), 1);
    for (int64_t i = kernel_shape.size() - 2; i >= 0; --i) {
      strides[i] = strides[i + 1] * kernel_shape[i + 1];
    }

    if (binary_inst->GetOpCode() == OpCode::MUL) {
      for (size_t i = 0, e = kernel_type.GetTotalNumOfElements(); i < e; ++i) {
        size_t ch = (i / strides[info.kernel_input_axis]) % in_chs;
        new_kernel_data[i] =
            kernel->GetDataAsFloat32(i) * c->GetDataAsFloat32(ch);
      }
      auto new_kernel =
          cb.CreateConstant(kernel->GetName(), kernel_type, new_kernel_data);
      operands[1] = *new_kernel;
      ret.second = *builder.Clone(*inst, operands);
    } else if (!has_padding && (inst->GetNumOfOperands() == 2 ||
                                (inst->GetNumOfOperands() == 3 &&
                                 IsA<Constant>(inst->GetOperand(2))))) {
      auto out_chs = kernel_shape[info.kernel_output_axis];
      // Conv(x + c), k) ==> Conv(x, k) + c * k)
      // C is vector (len = in_chs), reshape k to (H * W, out_chs, in_chs)
      std::vector<float> new_bias(out_chs);
      std::string bias_name = inst->GetName() + "_bias";
      if (inst->GetNumOfOperands() == 3) {
        auto orig_bias = DynCast<Constant>(inst->GetOperand(2));
        bias_name = orig_bias->GetName() + "_fused";
        HLCHECK(orig_bias->GetResultType().GetTotalNumOfElements() == out_chs);
        for (int i = 0; i < out_chs; ++i) {
          new_bias[i] = orig_bias->GetDataAsFloat32(i);
        }
      }
      auto hw = kernel_type.GetTotalNumOfElements() / in_chs / out_chs;
      auto stride_s = strides[info.kernel_width_axis];
      auto stride_o = strides[info.kernel_output_axis];
      auto stride_i = strides[info.kernel_input_axis];
      for (int s = 0; s < hw; ++s) {
        for (int och = 0; och < out_chs; ++och) {
          std::vector<float> m(in_chs);
          float t = 0;
          for (int i = 0; i < in_chs; ++i) {
            t += kernel->GetDataAsFloat32(s * stride_s + och * stride_o +
                                          i * stride_i) *
                 c->GetDataAsFloat32(i);
          }
          new_bias[och] += t;
        }
      }
      halo::Type type{kernel_type.GetDataType(), {out_chs}};

      auto new_bias_op = cb.CreateConstant(bias_name, type, new_bias);
      if (operands.size() >= 3) {
        operands[2] = *new_bias_op;
      } else {
        operands.push_back(*new_bias_op);
      }

      ret.second = *builder.Clone(*inst, operands);
    }
  }
  return ret;
}

static void Pad(char* dst, const char* src, size_t elems_num, size_t elem_size,
                const std::vector<int64_t>& orig_shape,
                const std::vector<int64_t>& new_shape,
                const std::vector<int32_t>& padding_amt) {
  std::vector<size_t> pos(orig_shape.size());
  std::vector<size_t> dst_strides(orig_shape.size(), 1);
  int dims = orig_shape.size();
  for (int i = dims - 2; i >= 0; --i) {
    dst_strides[i] = dst_strides[i + 1] * new_shape[i + 1];
  }

  for (size_t i = 0; i < elems_num; ++i) {
    auto dst_pos = pos;
    for (int j = 0; j < dims; ++j) {
      dst_pos[j] += padding_amt[j];
    }
    size_t dst_offset = std::inner_product(dst_pos.begin(), dst_pos.end(),
                                           dst_strides.begin(), 0UL) *
                        elem_size;

    std::copy(src, src + elem_size, dst + dst_offset); // NOLINT.
    src += elem_size;                                  // NOLINT.
    int c = 1;
    for (int j = dims - 1; j >= 0 && c == 1; --j) {
      pos[j] += c;
      if (pos[j] >= static_cast<size_t>(orig_shape[j])) {
        pos[j] = 0;
      } else {
        c = 0;
      }
    }
  }
}

std::pair<Def, Def> InstSimplify::RunOnInstruction(PadInst* pad_inst) {
  Def orig_def{pad_inst, 0};
  Def op0 = pad_inst->GetOperand(0);
  Def op1 = pad_inst->GetOperand(1);

  if (!IsA<Constant>(op0.GetOwner()) || !IsA<Constant>(op1.GetOwner()) ||
      pad_inst->GetNumOfOperands() != 2 ||
      pad_inst->GetMode() != PadMode::CONSTANT) {
    return {orig_def, orig_def};
  }

  const Constant* data = DynCast<Constant>(op0.GetOwner());
  const Constant* pad_amt = DynCast<Constant>(op1.GetOwner());

  ConstantBuilder cb(pad_inst->GetParent()->GetParent());
  auto dims = op0.GetType().GetNumOfDims();
  std::vector<int64_t> shape(dims);
  std::vector<int32_t> paddings_before(dims);
  for (size_t i = 0; i < dims; ++i) {
    int32_t before = pad_amt->GetDataPtr<int32_t>()[i * 2];    // NOLINT
    int32_t after = pad_amt->GetDataPtr<int32_t>()[i * 2 + 1]; // NOLINT
    shape[i] = op0.GetType().GetNumOfElementsInDim(i) + before + after;
    paddings_before[i] = before;
  }
  halo::Type type{op0.GetType().GetDataType(), shape};

  std::vector<char> w(type.GetTotalNumOfElements() *
                      data->GetElementSizeInBytes());
  std::vector<size_t> dim_sizes(dims);

  Pad(w.data(), static_cast<const char*>(data->GetRawDataPtr()),
      op0.GetType().GetTotalNumOfElements(), data->GetElementSizeInBytes(),
      op0.GetType().GetDimSizes(), shape, paddings_before);

  auto new_inst = cb.CreateConstant("folded_pad", type, w.data());
  return {orig_def, Def{new_inst, 0}};
}

std::pair<Def, Def> InstSimplify::RunOnInstruction(ReshapeInst* reshape_inst) {
  Def orig_def{reshape_inst, 0};
  // for reshape(reshape(x, c0), c1), replace it with reshape(x, c1).
  const auto& op0 = reshape_inst->GetOperand(0);
  if (IsA<ReshapeInst>(op0)) {
    IRBuilder builder(reshape_inst->GetParent());
    builder.SetInsertAfter(reshape_inst);
    auto new_inst = builder.CreateReshape(reshape_inst->GetName(),
                                          op0.GetDef()->GetOperand(0),
                                          reshape_inst->GetOperand(1));
    new_inst->GetResultsTypes()[0] = reshape_inst->GetResultsTypes()[0];
    return {orig_def, Def{new_inst, 0}};
  }

  const auto& input_type = op0.GetType();
  const auto& ret_type = reshape_inst->GetResultType();
  if (input_type.IsValid() && ret_type.IsValid() && input_type == ret_type) {
    return {orig_def, op0};
  }

  if (IsA<Constant>(op0) && reshape_inst->GetResultType().IsValid()) {
    Constant* src = DynCast<Constant>(op0);
    Constant* new_c = nullptr;
    if (op0.GetDef()->GetNumberOfUses() == 1) {
      new_c = src;
    } else {
      ConstantBuilder cb(reshape_inst->GetParent()->GetParent());
      new_c = cb.Clone(*src);
    }
    new_c->GetResultsTypes()[0] = reshape_inst->GetResultType();
    return {orig_def, *new_c};
  }

  return {orig_def, orig_def};
}

std::pair<Def, Def> InstSimplify::RunOnInstruction(ExpandDimsInst* inst) {
  HLCHECK(inst->GetNumOfOperands() == 2);
  auto input = inst->GetOperand(0);
  const auto& input_type = input.GetType();
  Def orig_def{inst, 0};

  if (!input_type.IsValid() || !IsA<Constant>(inst->GetOperand(1))) {
    return {orig_def, orig_def};
  }
  const Constant* shape = DynCast<Constant>(inst->GetOperand(1));
  auto input_elem = input_type.GetTotalNumOfElements();
  HLCHECK(shape->GetResultType().GetNumOfDims() == 1);
  std::vector<int64_t> output_shape;
  std::vector<int64_t> output_extends;

  IRBuilder builder(inst->GetParent());
  builder.SetInsertAfter(inst);

  int shape_rank = shape->GetResultType().GetTotalNumOfElements();
  int input_rank = input_type.GetNumOfDims();
  auto src_extends = GetExtends(input_type.GetDimSizes());
  for (int i = 0, e = std::max(shape_rank, input_rank); i < e; ++i) {
    int input_idx = input_rank - 1 - i;
    int shape_idx = shape_rank - 1 - i;
    int64_t dim0 =
        (input_idx < 0) ? 1 : input_type.GetNumOfElementsInDim(input_idx);
    int64_t dim1 = (shape_idx < 0) ? 1 : shape->GetDataAsInt64(shape_idx);
    HLCHECK(dim0 == dim1 || dim0 == 1 || dim1 == 1);
    output_shape.push_back((dim0 == 1) ? dim1 : dim0);
    bool is_bs = dim0 == 1;
    output_extends.push_back(is_bs ? 0 : src_extends[input_idx]);
  }
  std::reverse(output_shape.begin(), output_shape.end());
  std::reverse(output_extends.begin(), output_extends.end());

  halo::Type ret_type{input_type.GetDataType(), output_shape};
  auto ret_elem = ret_type.GetTotalNumOfElements();

  ConstantBuilder cb(inst->GetParent()->GetParent());
  if (input_elem == ret_elem) {
    Constant* c = cb.CreateConstant(
        inst->GetName() + "_expand",
        halo::Type{DataType::INT64,
                   {static_cast<int64_t>(output_shape.size())}},
        output_shape.data());
    auto reshape =
        builder.CreateReshape(inst->GetName(), {inst->GetOperand(0), *c});

    return {orig_def, *reshape};
  }
  if (IsA<Constant>(inst->GetOperand(0))) {
    const Constant* src = DynCast<Constant>(input);
    DefaultDataLayout data_layout;
    size_t elem_size = data_layout.Bytes(input_type.GetDataType());
    std::vector<unsigned char> buf(ret_elem * elem_size);
    const auto& dst_extends = GetExtends(output_shape);
    for (int64_t dst_idx = 0; dst_idx < ret_elem; ++dst_idx) {
      std::vector<int64_t> dst_dims(output_shape.size());
      for (int64_t i = 0, e = dst_dims.size(), t = dst_idx; t >= 0 && i < e;
           ++i) {
        dst_dims[i] = t / dst_extends[i];
        t -= dst_dims[i] * dst_extends[i];
      }
      auto src_idx = std::inner_product(
          output_extends.begin(), output_extends.end(), dst_dims.begin(), 0L);
      const unsigned char* src_ptr =
          static_cast<const unsigned char*>(src->GetRawDataPtr()) + // NOLINT.
          src_idx * elem_size;
      std::copy_n(src_ptr, elem_size, buf.begin() + dst_idx * elem_size);
    }
    Constant* c =
        cb.CreateConstant(inst->GetName() + "_expand", ret_type, buf.data());
    return {orig_def, *c};
  }
  return {orig_def, orig_def};
}

std::pair<Def, Def> InstSimplify::RunOnInstruction(BatchNormInst* inst) {
  Def orig_def{inst, 0};
  int num_inputs = inst->GetNumOfOperands();
  const auto& input_type = inst->GetResultType();

  auto input = inst->GetOperand(0);
  // Not profitable if the mul cannot be fused.
  bool is_profitable =
      IsA<Conv2DInst>(input) || IsA<Conv2DTransposeInst>(input);
  if (disable_conv_bn_ || !is_profitable || num_inputs <= 4 ||
      !input_type.IsValid() || input_type.GetNumOfDims() != 4 ||
      !IsA<Constant>(inst->GetOperand(3)) ||
      !IsA<Constant>(inst->GetOperand(4))) {
    return {orig_def, orig_def};
  }
  auto scale = DynCast<Constant>(inst->GetOperand(1));
  auto offset = DynCast<Constant>(inst->GetOperand(2));
  auto mean = DynCast<Constant>(inst->GetOperand(3));
  auto variance = DynCast<Constant>(inst->GetOperand(4));

  int ch_dim = inst->GetDataFormat() == DataFormat::NCHW ? 1 : 3;
  auto ch_num = input_type.GetNumOfElementsInDim(ch_dim);
  const auto& mean_type = mean->GetResultType();
  if (mean_type.GetTotalNumOfElements() != ch_num ||
      variance->GetResultType().GetTotalNumOfElements() != ch_num ||
      mean_type.GetDataType() != DataType::FLOAT32) {
    return {orig_def, orig_def};
  }

  std::vector<float> mul_buf(ch_num);
  std::vector<float> add_buf(ch_num);

  float eps = inst->GetEpsilon();
  // Convert BN to Ax + B.
  for (int64_t i = 0; i < ch_num; ++i) {
    mul_buf[i] = 1.0F / std::sqrt(variance->GetData<float>(i) + eps);
    float s = scale->GetData<float>(i);
    mul_buf[i] *= s;
    float b = offset->GetData<float>(i);
    add_buf[i] = -mean->GetData<float>(i) * mul_buf[i] + b;
  }
  std::vector<int64_t> shape(input_type.GetNumOfDims(), 1);
  shape[ch_dim] = ch_num;
  halo::Type ty{mean_type.GetDataType(), shape};
  ConstantBuilder cb(inst->GetParent()->GetParent());
  auto new_scale =
      cb.CreateConstant(inst->GetName() + "_0", ty, mul_buf.data());
  auto new_offset =
      cb.CreateConstant(inst->GetName() + "_1", ty, add_buf.data());
  IRBuilder builder(inst->GetParent());
  builder.SetInsertAfter(inst);
  auto new_mul = builder.CreateMul(inst->GetName() + "_mul", input, *new_scale);
  auto new_add =
      builder.CreateAdd(inst->GetName() + "_add", *new_mul, *new_offset);
  return {orig_def, *new_add};
}

std::pair<Def, Def> InstSimplify::RunOnInstruction(StackInst* inst) {
  Def orig_def{inst, 0};
  int num_inputs = inst->GetNumOfOperands();
  if (inst->GetAxis() != 0) {
    return {orig_def, orig_def};
  }
  for (int i = 0; i < num_inputs; ++i) {
    if (!IsA<Constant>(inst->GetOperand(i))) {
      return {orig_def, orig_def};
    }
  }
  // convert to an array of constant
  const auto& input0_type = inst->GetOperand(0).GetType();
  std::vector<int64_t> ret_shape(input0_type.GetDimSizes());
  ret_shape.insert(ret_shape.begin(), num_inputs);
  ConstantBuilder cb(inst->GetParent()->GetParent());
  auto count = input0_type.GetTotalNumOfElements();
  Constant* c_input0 = DynCast<Constant>(inst->GetOperand(0).GetOwner());
  size_t copy_size = count * c_input0->GetElementSizeInBytes();
  std::vector<char> buf(num_inputs * copy_size);
  for (int i = 0; i < num_inputs; ++i) {
    Constant* c_input_i = DynCast<Constant>(inst->GetOperand(i).GetOwner());
    memcpy(&buf[copy_size * i], c_input_i->GetRawDataPtr(), copy_size);
  }
  halo::Type result_type{input0_type.GetDataType(), ret_shape};
  auto new_def =
      cb.CreateConstant(inst->GetName() + "_folding", result_type, buf.data());
  return {orig_def, *new_def};
}

std::pair<Def, Def> InstSimplify::RunOnInstruction(ZExtInst* inst) {
  Def orig_def{inst, 0};
  DataType ret_dt = inst->GetDataType();
  auto op0 = inst->GetOperand(0);
  const auto& op0_type = op0.GetType();
  DataType src_dt = op0_type.GetDataType();
  HLCHECK(halo::Type::IsIntegerType(src_dt) &&
          halo::Type::IsIntegerType(ret_dt));

  if (!op0_type.IsValid() || !IsA<Constant>(op0)) {
    return {orig_def, orig_def};
  }

  ConstantBuilder cb(inst->GetParent()->GetParent());
  Constant* c_src = DynCast<Constant>(op0.GetOwner());
  if (ret_dt == DataType::INT32 && src_dt == DataType::BOOL) {
    std::vector<int> ret;
    ret.reserve(op0_type.GetTotalNumOfElements());
    for (int i = 0; i < op0_type.GetTotalNumOfElements(); ++i) {
      ret.push_back(static_cast<int>(c_src->GetData<int8_t>(i)));
    }
    Constant* c_ret = cb.CreateConstant(
        inst->GetName(), halo::Type{ret_dt, op0_type.GetDimSizes()},
        ret.data());
    return {orig_def, *c_ret};
  }

  if (ret_dt == DataType::INT32 && src_dt == DataType::INT64) {
    std::vector<int32_t> ret;
    ret.reserve(op0_type.GetTotalNumOfElements());
    for (int i = 0, e = op0_type.GetTotalNumOfElements(); i != e; ++i) {
      ret.push_back(static_cast<int32_t>(c_src->GetData<int64_t>(i)));
    }
    Constant* c_ret = cb.CreateConstant(
        inst->GetName() + "_folding",
        halo::Type{ret_dt, op0_type.GetDimSizes()}, ret.data());
    return {orig_def, *c_ret};
  }

  if (ret_dt == DataType::INT64 && src_dt == DataType::INT32) {
    std::vector<int64_t> ret;
    ret.reserve(op0_type.GetTotalNumOfElements());
    for (int i = 0, e = op0_type.GetTotalNumOfElements(); i != e; ++i) {
      ret.push_back(static_cast<int64_t>(c_src->GetData<int32_t>(i)));
    }
    Constant* c_ret = cb.CreateConstant(
        inst->GetName() + "_folding",
        halo::Type{ret_dt, op0_type.GetDimSizes()}, ret.data());
    return {orig_def, *c_ret};
  }
  return {orig_def, orig_def};
}

std::pair<Def, Def> InstSimplify::RunOnInstruction(RangeInst* inst) {
  Def orig_def{inst, 0};
  auto op0 = inst->GetOperand(0);
  auto op1 = inst->GetOperand(1);
  auto op2 = inst->GetOperand(2);
  DataType dt = op0.GetType().GetDataType();

  if (!IsA<Constant>(op0.GetOwner()) || !IsA<Constant>(op1.GetOwner()) ||
      !IsA<Constant>(op2.GetOwner()) || dt != DataType::INT32) {
    return {orig_def, orig_def};
  }

  int64_t num_elements = 0;
  Constant* c_op0 = DynCast<Constant>(op0.GetOwner());
  Constant* c_op1 = DynCast<Constant>(op1.GetOwner());
  Constant* c_op2 = DynCast<Constant>(op2.GetOwner());

  int begin = c_op0->GetData<int32_t>(0);
  int end = c_op1->GetData<int32_t>(0);
  int step = c_op2->GetData<int32_t>(0);
  num_elements = std::max(0, (end - begin) / step);
  HLCHECK(num_elements);
  std::vector<int> ret_data(num_elements);
  ret_data[0] = begin;
  for (int i = 1; i < num_elements; ++i) {
    ret_data[i] = ret_data[i - 1] + step;
  }
  ConstantBuilder cb(inst->GetParent()->GetParent());
  Constant* c_ret =
      cb.CreateConstant(inst->GetName() + "_folding",
                        halo::Type{dt, {num_elements}}, ret_data.data());
  return {orig_def, *c_ret};
}

std::pair<Def, Def> InstSimplify::RunOnInstruction(SetDiff1DInst* inst) {
  Def orig_def{inst, 0};
  auto op0 = inst->GetOperand(0);
  auto op1 = inst->GetOperand(1);
  const auto& op0_type = op0.GetType();
  const auto& op1_type = op1.GetType();
  DataType dt = op0_type.GetDataType();
  if (!IsA<Constant>(op0.GetOwner()) || !IsA<Constant>(op1.GetOwner()) ||
      dt != DataType::INT32) {
    return {orig_def, orig_def};
  }
  std::vector<int> ret_data;
  std::unordered_set<int> diff_set;
  Constant* c_op0 = DynCast<Constant>(op0.GetOwner());
  Constant* c_op1 = DynCast<Constant>(op1.GetOwner());
  for (int i = 0, e = op1_type.GetTotalNumOfElements(); i != e; ++i) {
    diff_set.emplace(c_op1->GetData<int>(i));
  }
  for (int i = 0, e = op0_type.GetTotalNumOfElements(); i != e; ++i) {
    int data_i = c_op0->GetData<int>(i);
    if (diff_set.count(data_i) == 0) {
      ret_data.push_back(data_i);
    }
  }
  ConstantBuilder cb(inst->GetParent()->GetParent());
  Constant* c_ret = cb.CreateConstant(
      inst->GetName() + "_folding",
      halo::Type{dt, {static_cast<int64_t>(ret_data.size())}}, ret_data.data());
  return {orig_def, *c_ret};
}

std::pair<Def, Def> InstSimplify::RunOnInstruction(GatherInst* inst) {
  Def orig_def{inst, 0};
  int axis = inst->GetAxis();
  const auto& dst_type = inst->GetResultsTypes()[0];
  const auto& type_op0 = inst->GetOperand(0).GetType();
  const auto& op1 = inst->GetOperand(1);
  // Gather(data, ZExt(index, int64)) ==> Gather(data, index)
  if (IsA<ZExtInst>(op1)) {
    IRBuilder builder(inst->GetParent());
    ZExtInst* zext = DynCast<ZExtInst>(op1.GetDef());
    builder.SetInsertAfter(inst);
    auto ops = inst->GetOperands();
    ops[1] = zext->GetOperand(0);
    auto new_inst = builder.Clone(*inst, ops);
    return {orig_def, *new_inst};
  }

  if (!type_op0.IsValid()) {
    return {orig_def, orig_def};
  }
  if (axis < 0) {
    axis += type_op0.GetNumOfDims();
  }
  HLCHECK(axis >= 0 && axis < static_cast<int>(type_op0.GetNumOfDims()));

  for (size_t i = 0; i < inst->GetNumOfOperands(); ++i) {
    if (!IsA<Constant>(inst->GetOperand(i).GetOwner())) {
      return {orig_def, orig_def};
    }
  }

  if (!dst_type.IsValid()) {
    return {orig_def, orig_def};
  }

  Constant* c_op0 = DynCast<Constant>(inst->GetOperand(0).GetOwner());
  const auto& type_op1 = inst->GetOperand(1).GetType();
  Constant* c_op1 = DynCast<Constant>(inst->GetOperand(1).GetOwner());
  DataType dt = dst_type.GetDataType();
  DefaultDataLayout data_layout;

  size_t byte_per_element = data_layout.Bytes(dt);
  size_t bytes = byte_per_element * dst_type.GetTotalNumOfElements();
  std::vector<unsigned char> buf(bytes);
  size_t per_copy_bytes = byte_per_element;
  auto dst_extends = GetExtends(dst_type.GetDimSizes());
  auto src_extends = GetExtends(type_op0.GetDimSizes());
  auto idx_extends = GetExtends(type_op1.GetDimSizes());
  int64_t dst_rank = dst_type.GetNumOfDims();
  if (dst_rank == 0 && dst_type.GetTotalNumOfElements() == 1) {
    dst_rank = 1;
  }
  int op1_rank = type_op1.GetNumOfDims();
  if (op1_rank == 0 && type_op1.GetTotalNumOfElements() == 1) {
    op1_rank = 1;
  }
  for (int64_t dst_idx = 0, e = dst_type.GetTotalNumOfElements(); dst_idx < e;
       ++dst_idx) {
    // map dst_idx to src_idx.
    std::vector<int64_t> dst_dims(dst_rank);
    for (int64_t i = 0, k = dst_idx; k > 0 && i < dst_rank; ++i) {
      dst_dims[i] = k / dst_extends[i];
      k -= dst_dims[i] * dst_extends[i];
    }
    std::vector<int64_t> src_dims(type_op0.GetNumOfDims());
    for (int i = 0; i < dst_rank; ++i) {
      if (i < axis) {
        src_dims[i] = dst_dims[i];
      } else if (i >= (axis + op1_rank)) {
        src_dims[i - op1_rank + 1] = dst_dims[i];
      } else {
        int64_t idx = std::inner_product(idx_extends.begin(), idx_extends.end(),
                                         dst_dims.begin() + axis, 0L);
        src_dims[axis] = c_op1->GetDataAsInt64(idx);
      }
    }
    int64_t src_idx = std::inner_product(src_dims.begin(), src_dims.end(),
                                         src_extends.begin(), 0L);
    unsigned char* src_ptr =
        static_cast<unsigned char*>(c_op0->GetRawDataPtr()) + // NOLINT.
        src_idx * per_copy_bytes;
    std::copy_n(src_ptr, per_copy_bytes,
                buf.begin() + dst_idx * per_copy_bytes);
  }
  ConstantBuilder cb(inst->GetParent()->GetParent());
  Constant* c_ret =
      cb.CreateConstant(inst->GetName() + "_folding", dst_type, buf.data());
  return {orig_def, *c_ret};
}

std::pair<Def, Def> InstSimplify::RunOnInstruction(CeilInst* inst) {
  Def orig_def{inst, 0};
  const auto& op0 = inst->GetOperand(0);
  if (IsA<Constant>(op0)) {
    const auto c = DynCast<Constant>(op0);
    ConstantBuilder cb(inst->GetParent()->GetParent());
    const auto& dt = op0.GetType();
    if (dt.GetDataType() == DataType::FLOAT32) {
      std::vector<float> ceiled(dt.GetTotalNumOfElements());
      for (int i = 0, e = ceiled.size(); i < e; ++i) {
        ceiled[i] = std::ceil(c->GetDataAsFloat32(i));
      }
      return {orig_def,
              *cb.CreateConstant(c->GetName() + "_ceil", dt, ceiled.data())};
    }
  }
  return {orig_def, orig_def};
}

std::pair<Def, Def> InstSimplify::RunOnInstruction(ConcatInst* inst) {
  Def orig_def{inst, 0};
  // Concat(Transpose(A), Transpose(B),...) => Transpose(Concat(A, B))
  std::vector<int> perm;
  size_t n = inst->GetNumOfOperands();
  std::vector<Def> tr_ops;
  tr_ops.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    auto op = inst->GetOperand(i);
    if (!IsA<TransposeInst>(op)) {
      break;
    }
    const TransposeInst* tr = DynCast<const TransposeInst>(op);
    if (i == 0) {
      perm = tr->GetPermutation();
    } else if (perm != tr->GetPermutation()) {
      break;
    }
    tr_ops.push_back(tr->GetOperand(0));
  }
  if (tr_ops.size() == n) {
    IRBuilder builder(inst->GetParent());
    builder.SetInsertAfter(inst);
    auto new_concat = builder.CreateConcat(inst->GetName(), tr_ops);
    new_concat->SetAxis(perm[inst->GetAxis()]);
    TransposeInst* new_tr =
        builder.CreateTranspose(new_concat->GetName() + "_t", {*new_concat});
    new_tr->SetPermutation(perm);
    return {orig_def, *new_tr};
  }

  for (size_t i = 0; i < inst->GetNumOfOperands(); ++i) {
    if (!IsA<Constant>(inst->GetOperand(i).GetOwner())) {
      return {orig_def, orig_def};
    }
  }

  int num_inputs = inst->GetN();
  int axis = inst->GetAxis();
  const auto& dst_type = inst->GetResultsTypes()[0];
  if (!dst_type.IsValid() || axis != 0) {
    return {orig_def, orig_def};
  }
  // Constant propagating on axis = 0
  DataType dt = dst_type.GetDataType();
  DefaultDataLayout data_layout;

  size_t byte_per_element = data_layout.Bytes(dt);
  size_t bytes = byte_per_element * dst_type.GetTotalNumOfElements();
  std::vector<unsigned char> buf(bytes);
  size_t offset = 0;
  for (int i = 0; i < num_inputs; ++i) {
    auto input = inst->GetOperand(i);
    Constant* c_input = DynCast<Constant>(input.GetOwner());
    size_t num_elements = input.GetType().GetTotalNumOfElements();
    size_t copy_bytes = num_elements * byte_per_element;
    std::copy_n(static_cast<unsigned char*>(c_input->GetRawDataPtr()),
                copy_bytes, buf.begin() + offset);
    offset += copy_bytes;
  }
  ConstantBuilder cb(inst->GetParent()->GetParent());
  Constant* c_ret =
      cb.CreateConstant(inst->GetName() + "_folding", dst_type, buf.data());
  return {orig_def, *c_ret};
}

std::pair<Def, Def> InstSimplify::RunOnInstruction(TransposeInst* inst) {
  Def orig_def{inst, 0};
  std::pair<Def, Def> ret{orig_def, orig_def};

  const auto& input = inst->GetOperand(0);

  // Fold constant perm op into attribute.
  if (inst->GetNumOfOperands() == 2 && IsA<Constant>(inst->GetOperand(1))) {
    const Constant* data = DynCast<Constant>(inst->GetOperand(1));
    const auto& ty = data->GetResultType();
    if (ty.GetDataType() == DataType::INT32) {
      const int32_t* ptr = data->GetDataPtr<int32_t>();
      std::vector<int> perms(ptr, ptr + ty.GetTotalNumOfElements()); // NOLINT
      IRBuilder builder(inst->GetParent());
      builder.SetInsertAfter(inst);
      TransposeInst* new_inst =
          builder.CreateTranspose(inst->GetName(), {input});
      new_inst->SetPermutation(perms);
      ret.second = *new_inst;
      return ret;
    }
  }

  const auto& perm = inst->GetPermutation();
  auto input_type = (input.GetDef()->GetResultsTypes()[input.GetIdx()]);
  int dims = -1;
  std::vector<int64_t> new_shape;
  std::vector<size_t> perm_strides;
  std::vector<size_t> orig_strides;
  if (input_type.IsValid()) {
    const auto& orig_shape = input_type.GetDimSizes();
    dims = orig_shape.size();
    HLCHECK(orig_shape.size() == inst->GetPermutation().size());
    orig_strides = std::vector<size_t>(dims, 1);
    for (int i = dims - 2; i >= 0; --i) {
      orig_strides[i] = orig_strides[i + 1] * orig_shape[i + 1];
    }
    new_shape = orig_shape;
    perm_strides = orig_strides;
    for (int i = 0; i < dims; ++i) {
      new_shape[i] = orig_shape[perm[i]];
      perm_strides[i] = orig_strides[perm[i]];
    }
  }

  if (IsA<Constant>(input)) {
    // Do transpose at compile time.
    auto orig = DynCast<Constant>(input.GetOwner());
    ConstantBuilder cb(inst->GetParent()->GetParent());
    const auto& type = orig->GetResultType();
    std::vector<int> pos(dims); // tracks the position of dst tensor.

    size_t elem_size = orig->GetElementSizeInBytes();
    size_t elem_cnt = type.GetTotalNumOfElements();
    std::vector<char> buf(elem_size * elem_cnt);
    for (size_t i = 0; i < elem_cnt; ++i) {
      size_t offset = std::inner_product(pos.begin(), pos.end(),
                                         perm_strides.begin(), 0UL) *
                      elem_size;
      const char* src =
          static_cast<const char*>(orig->GetRawDataPtr()) + offset; // NOLINT
      memcpy(&buf[i * elem_size], src, elem_size);
      int c = 1;
      for (int i = dims - 1; i >= 0 && c == 1; --i) {
        pos[i] += c;
        if (pos[i] >= new_shape[i]) {
          pos[i] = 0;
        } else {
          c = 0;
        }
      }
    }
    auto new_c = cb.CreateConstant(orig->GetName() + "_T",
                                   halo::Type{type.GetDataType(), new_shape},
                                   buf.data());
    ret.second = *new_c;
    return ret;
  }

  if (remove_input_transpose_ && input.GetUses().size() == 1) {
    if (IsA<Argument>(input)) {
      Argument* arg = DynCast<Argument>(input);
      const auto& orig_dims = input.GetType().GetDimSizes();
      const auto& perms = inst->GetPermutation();
      auto new_dims = orig_dims;
      for (int i = 0, e = orig_dims.size(); i < e; ++i) {
        new_dims[i] = orig_dims[perms[i]];
      }
      halo::Type ty{input.GetType().GetDataType(), new_dims};
      arg->SetType(ty);
      ret.second = input;
      return ret;
    }
  }

  // Transpose(Transpose(in, perm0), perm1) => Transpose(in, perm2)
  if (IsA<TransposeInst>(input)) {
    const TransposeInst* t0 = DynCast<TransposeInst>(input.GetOwner());
    const auto& perm0 = t0->GetPermutation();
    HLCHECK(perm0.size() == perm.size());
    auto new_perm = perm0;
    for (int i = 0, e = perm0.size(); i < e; ++i) {
      new_perm[i] = perm[perm0[i]];
    }
    IRBuilder builder(inst->GetParent());
    builder.SetInsertAfter(inst);
    TransposeInst* new_trans =
        builder.CreateTranspose(inst->GetName(), {t0->GetOperand(0)});
    new_trans->SetPermutation(new_perm);
    ret.second = *new_trans;
    return ret;
  }

  // Check if it is a redundant permute (all other dims are 1)
  const auto& type = input.GetType();
  const auto& out_type = inst->GetResultType();
  if (type.IsValid() && out_type.IsValid()) {
    unsigned non_ones = 0;
    for (auto& d : type.GetDimSizes()) {
      non_ones += d == 1 ? 0 : 1;
    }
    if (non_ones == 1) {
      IRBuilder builder(inst->GetParent());
      builder.SetInsertAfter(inst);
      ConstantBuilder cb(inst->GetParent()->GetParent());
      halo::Type reshape_ty{DataType::INT64,
                            {static_cast<int64_t>(out_type.GetNumOfDims())}};
      auto shape = cb.CreateConstant(inst->GetName() + "_reshape", reshape_ty,
                                     out_type.GetDimSizes().data());
      ReshapeInst* reshape =
          builder.CreateReshape(inst->GetName(), {input, *shape});
      reshape->GetResultsTypes()[0] = out_type;
      ret.second = *reshape;
      return ret;
    }
  }

  // Check if a permutaton is redundant.
  bool is_redundant = true;
  for (int i = 0, e = perm.size(); i < e; ++i) {
    if (perm[i] != i) {
      is_redundant = false;
      break;
    }
  }

  if (is_redundant) {
    ret.second = inst->GetOperand(0);
    return ret;
  }

  // Transpose => Reshape if its' like (N, 1, H, W) => (N, H, W, 1)
  if (dims > 0) {
    const auto& orig_shape = input_type.GetDimSizes();
    auto new_perm = perm;
    for (int i = 0, a = 0; i < dims; ++i, ++a) {
      if (orig_shape[i] == 1) {
        for (auto& v : new_perm) {
          if (v == a) {
            v = -1; // skip the dim
          }
          if (v > a) {
            --v;
          }
        }
        --a;
      }
    }

    bool reshapable = true;
    for (int i = 0, c = 0; i < dims && reshapable; ++i) {
      if (new_perm[i] >= 0) {
        if (new_perm[i] != c++) {
          reshapable = false;
        }
      }
    }
    if (reshapable) {
      IRBuilder builder(inst->GetParent());
      builder.SetInsertAfter(inst);
      ConstantBuilder cb(inst->GetParent()->GetParent());
      Constant* c_shape = cb.CreateConstant(
          inst->GetName() + "_shape",
          halo::Type{DataType::INT64, std::vector<int64_t>{dims}},
          new_shape.data());
      auto reshape = builder.CreateReshape(inst->GetName(),
                                           {inst->GetOperand(0), *c_shape});
      reshape->GetResultsTypes()[0] = inst->GetResultType();
      ret.second = *reshape;
      return ret;
    }
  }
  return ret;
}

std::pair<Def, Def> InstSimplify::RunOnInstruction(ReturnInst* inst) {
  Def orig_def{inst, 0};
  if (!remove_output_transpose_) {
    return {orig_def, orig_def};
  }
  for (int i = 0, e = inst->GetNumOfOperands(); i < e; ++i) {
    const auto& op = inst->GetOperand(i);
    if (IsA<TransposeInst>(op)) {
      inst->ReplaceOperandWith(i, DynCast<Instruction>(op)->GetOperand(0));
    }
  }
  return {orig_def, orig_def};
}

std::pair<Def, Def> InstSimplify::RunOnInstruction(RandomUniformInst* inst) {
  Def orig_def{inst, 0};
  const auto& dst_type = inst->GetResultsTypes()[0];
  if (dst_type.IsValid() && dst_type.GetDataType() == DataType::FLOAT32) {
    auto noe = dst_type.GetTotalNumOfElements();
    float max_val = inst->GetMaxval();
    float min_val = inst->GetMinval();
    // int seed = inst->GetSeed();
    std::vector<float> ret(noe);
    // std::random_device rd;
    // std::mt19937 gen(rd());
    std::mt19937 gen(1234); // NOLINT
    std::uniform_real_distribution<> dis(min_val, max_val);
    for (int i = 0; i < noe; ++i) {
      ret[i] = dis(gen);
    }
    ConstantBuilder cb(inst->GetParent()->GetParent());
    Constant* c_ret =
        cb.CreateConstant(inst->GetName() + "_folding", dst_type, ret.data());
    return {orig_def, *c_ret};
  }
  return {orig_def, orig_def};
}

std::pair<Def, Def> InstSimplify::RunOnInstruction(SliceInst* inst) {
  Def orig_def{inst, 0};
  auto op_len = inst->GetOperand(2);
  const auto& dst_type = inst->GetResultsTypes()[0];
  const auto& src_type = inst->GetOperand(0).GetType();
  if (dst_type.IsValid() && IsA<Constant>(op_len)) {
    Constant* c_size = DynCast<Constant>(op_len);
    if (op_len.GetType().GetDataType() == DataType::INT32) {
      int dim = op_len.GetType().GetTotalNumOfElements();
      std::vector<int> size_adj(dim);
      bool new_size = false;
      for (int i = 0; i != dim; ++i) {
        int size_i = c_size->GetData<int>(i);
        if (size_i == -1) {
          size_adj[i] = src_type.GetNumOfElementsInDim(i);
          new_size = true;
        } else {
          size_adj[i] = size_i;
        }
      }
      if (new_size) {
        ConstantBuilder cb(inst->GetParent()->GetParent());
        Constant* c_new_size =
            cb.CreateConstant(op_len.GetOwner()->GetName() + "_adj",
                              op_len.GetType(), size_adj.data());
        IRBuilder builder(inst->GetParent());
        builder.SetInsertAfter(inst);
        SliceInst* new_inst = builder.CreateSlice(
            inst->GetName(),
            {inst->GetOperand(0), inst->GetOperand(1), *c_new_size});
        new_inst->GetResultsTypes()[0] = dst_type;
        return {orig_def, *new_inst};
      }
    }
  }
  const auto& op0 = inst->GetOperand(0);
  const auto& op_start = inst->GetOperand(1);

  bool has_constant_steps =
      (inst->GetNumOfOperands() < 4 || IsA<Constant>(inst->GetOperand(3)));
  bool has_constant_axes =
      (inst->GetNumOfOperands() <= 4 || IsA<Constant>(inst->GetOperand(4)));

  if (IsA<Constant>(op0) && IsA<Constant>(op_start) && IsA<Constant>(op_len) &&
      inst->GetResultType().IsValid() && has_constant_steps &&
      has_constant_axes) {
    Constant* input = DynCast<Constant>(op0);
    const auto& dt = inst->GetResultType();
    auto starts_c = DynCast<Constant>(op_start);
    auto lens_c = DynCast<Constant>(op_len);
    const auto& op0_type = op0.GetType();
    auto rank = op0_type.GetNumOfDims();
    std::unordered_set<int> axes;
    if (inst->GetNumOfOperands() > 4) {
      const auto& data =
          DynCast<Constant>(inst->GetOperand(4))->GetDataAsInt64();
      for (auto x : data) {
        axes.insert(x);
      }
    } else {
      for (size_t i = 0; i < rank; ++i) {
        axes.insert(i);
      }
    }
    std::vector<int> starts(rank);
    std::vector<int> lens(rank);
    std::vector<int> steps(rank);
    Constant* steps_c =
        has_constant_steps ? DynCast<Constant>(inst->GetOperand(3)) : nullptr;
    for (size_t axis = 0, k = 0; axis < rank; ++axis) {
      bool to_slice = has_constant_axes && axes.count(axis) != 0;
      starts[axis] = to_slice ? starts_c->GetDataAsInt64(k) : 0;
      lens[axis] = to_slice ? lens_c->GetDataAsInt64(k)
                            : op0_type.GetNumOfElementsInDim(axis);
      steps[axis] = to_slice ? steps_c->GetDataAsInt64(k) : 1L;
      k += to_slice ? 1 : 0;
    }

    if (rank == 1) {
      DefaultDataLayout dl;
      auto bytes = dl.DataLayout::Bytes(dt);
      auto es = dl.Bytes(dt.GetDataType());
      std::vector<char> data(bytes);
      int step = steps[0];
      auto len = lens[0];
      auto start = starts[0];
      const char* src = static_cast<const char*>(input->GetRawDataPtr());
      char* dst = data.data();
      HLCHECK(len == dt.GetTotalNumOfElements());
      for (int i = 0; i < len; ++i) {
        auto src_idx = start + i * step;
        HLCHECK(src_idx >= 0 &&
                src_idx < op0.GetType().GetTotalNumOfElements());
        std::memcpy(&dst[i * es], &src[src_idx * es], es); // NOLINT
      }
      ConstantBuilder cb(inst->GetParent()->GetParent());
      auto c = cb.CreateConstant(inst->GetName(), dt, data.data());
      return {orig_def, *c};
    }
  }

  return {orig_def, orig_def};
}

std::pair<Def, Def> InstSimplify::RunOnInstruction(FPtoSIInst* inst) {
  Def orig_def{inst, 0};
  auto op0 = inst->GetOperand(0);
  if (IsA<Constant>(op0)) {
    auto src_ty = op0.GetType().GetDataType();
    auto dst_ty = inst->GetResultType().GetDataType();
    ConstantBuilder cb(inst->GetParent()->GetParent());
    Constant* input = DynCast<Constant>(op0);
    auto n = op0.GetType().GetTotalNumOfElements();
    if (src_ty == DataType::FLOAT32 &&
        (dst_ty == DataType::INT32 || dst_ty == DataType::INT64 ||
         dst_ty == DataType::UINT32)) {
      Constant* c = nullptr;
      if (dst_ty == DataType::INT32) {
        std::vector<int32_t> vals(n);
        for (int64_t i = 0; i < n; ++i) {
          vals[i] = input->GetData<float>(i);
        }
        c = cb.CreateConstant(inst->GetName(), inst->GetResultType(),
                              vals.data());
      } else if (dst_ty == DataType::UINT32) {
        std::vector<uint32_t> vals(n);
        for (int64_t i = 0; i < n; ++i) {
          vals[i] = input->GetData<float>(i);
        }
        c = cb.CreateConstant(inst->GetName(), inst->GetResultType(),
                              vals.data());
      } else {
        std::vector<int64_t> vals(n);
        for (int64_t i = 0; i < n; ++i) {
          vals[i] = input->GetData<float>(i);
        }
        c = cb.CreateConstant(inst->GetName(), inst->GetResultType(),
                              vals.data());
      }
      return {orig_def, *c};
    }
  }
  return {orig_def, orig_def};
}

std::pair<Def, Def> InstSimplify::RunOnInstruction(SItoFPInst* inst) {
  Def orig_def{inst, 0};
  const auto& op0 = inst->GetOperand(0);

  if (IsA<ReshapeInst>(op0)) {
    auto* reshape_inst = DynCast<ReshapeInst>(op0);
    if (reshape_inst->GetNumberOfUses() == 1) {
      const auto& op_reshape = reshape_inst->GetOperand(0);
      if (IsA<Argument>(op_reshape.GetOwner())) {
        Argument* arg = DynCast<Argument>(op_reshape);
        if (arg->GetNumberOfUses() == 1 && op_reshape.GetType().IsValid()) {
          arg->SetType(halo::Type{DataType::FLOAT32,
                                  op_reshape.GetType().GetDimSizes()});
          return {orig_def, *reshape_inst};
        }
      }
    }
  } else if (IsA<Argument>(op0.GetOwner()) && op0.GetType().IsValid() &&
             DynCast<Argument>(op0.GetOwner())->GetNumberOfUses() == 1) {
    Argument* arg = DynCast<Argument>(op0.GetOwner());
    arg->SetType(halo::Type{DataType::FLOAT32, op0.GetType().GetDimSizes()});
    return {orig_def, *arg};
  } else if (IsA<Constant>(op0)) {
    auto src_ty = op0.GetType().GetDataType();
    Constant* input = DynCast<Constant>(op0);
    auto n = op0.GetType().GetTotalNumOfElements();
    if (inst->GetDataType() == DataType::FLOAT32 &&
        (src_ty == DataType::INT32 || src_ty == DataType::INT64)) {
      std::vector<float> vals(n);
      for (int64_t i = 0; i < n; ++i) {
        vals[i] = (src_ty == DataType::INT32) ? input->GetData<int32_t>(i)
                                              : input->GetData<int64_t>(i);
      }
      ConstantBuilder cb(inst->GetParent()->GetParent());
      auto c = cb.CreateConstant(inst->GetName(), inst->GetResultType(),
                                 vals.data());
      return {orig_def, *c};
    }
  }
  return {orig_def, orig_def};
}

std::pair<Def, Def> InstSimplify::RunOnInstruction(OneHotInst* inst) {
  Def orig_def{inst, 0};
  // work around on cxx target when backend doens't support onehot.
  if (!simplify_for_preprocess_) {
    return {orig_def, orig_def};
  }
  auto on_value = inst->GetOperand(2);
  auto off_value = inst->GetOperand(3);
  const auto& dst_type = inst->GetResultsTypes()[0];
  if (!IsA<Constant>(on_value.GetOwner()) ||
      !IsA<Constant>(off_value.GetOwner()) || !dst_type.IsValid()) {
    return {orig_def, orig_def};
  }

  const auto& op0 = inst->GetOperand(0);
  if (IsA<ReshapeInst>(op0)) {
    auto reshape_inst = DynCast<ReshapeInst>(op0);
    if (reshape_inst->GetNumberOfUses() == 1) {
      const auto& op_reshape = reshape_inst->GetOperand(0);
      if (IsA<Argument>(op_reshape)) {
        Argument* arg = DynCast<Argument>(op_reshape);
        if (arg->GetNumberOfUses() == 1 && op_reshape.GetType().IsValid()) {
          arg->SetType(halo::Type{on_value.GetType().GetDataType(),
                                  dst_type.GetDimSizes()});
          return {orig_def, *arg};
        }
      }
    }
  } else if (IsA<Argument>(op0.GetOwner()) && op0.GetType().IsValid() &&
             DynCast<Argument>(op0.GetOwner())->GetNumberOfUses() == 1) {
    Argument* arg = DynCast<Argument>(op0.GetOwner());
    arg->SetType(
        halo::Type{on_value.GetType().GetDataType(), dst_type.GetDimSizes()});
    return {orig_def, *arg};
  }
  return {orig_def, orig_def};
}

bool InstSimplify::RunOnBasicBlock(BasicBlock* bb) {
  bool changed = false;
  for (auto& inst_t : *bb) {
    Instruction* inst = inst_t.get();
    if (inst->GetNumberOfUses() == 0) {
      if (inst->GetOpCode() == OpCode::RETURN) {
        RunOnInstruction(DynCast<ReturnInst>(inst));
      }
      continue;
    }
    std::pair<Def, Def> ret{Def{inst, 0}, Def{inst, 0}};
    switch (inst->GetOpCode()) {
#define GET_INST_DOWNCAST_SWITCH_WITH_RETURN
#include "halo/lib/ir/instructions_info.def"
#undef GET_INST_DOWNCAST_SWITCH_WITH_RETURN
      default: {
        // skip extension instruction.
        continue;
      }
    }
    if (ret.first != ret.second) {
      changed |= true;
      if (ret.second.GetOwner() != nullptr) {
        // Replace all uses
        inst->ReplaceAllUsesWith(ret.first.GetIdx(), ret.second);
      }
    }
  }
  return changed;
}

} // end namespace halo
