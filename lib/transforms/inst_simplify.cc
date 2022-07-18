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
#include <cmath>
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
  Constant* c_lhs = DynCast<Constant>(op0);
  Constant* c_rhs = DynCast<Constant>(op1);
  if (c_lhs == nullptr || c_rhs == nullptr) {
    return nullptr;
  }

  if (opcode == OpCode::CMP) {
    if (pred == KindPredicate::GE || pred == KindPredicate::GT) {
      pred =
          (pred == KindPredicate::GE) ? KindPredicate::LT : KindPredicate::LE;
      std::swap(op0, op1);
      std::swap(c_lhs, c_rhs);
    }
  }

  ConstantAccessor<T> lhs_accessor(*c_lhs, ret_type);
  ConstantAccessor<T> rhs_accessor(*c_rhs, ret_type);
  auto lhs_it = lhs_accessor.begin();
  auto rhs_it = rhs_accessor.begin();

  size_t num_elements = ret_type.GetTotalNumOfElements();
  Constant* c_ret = nullptr;
  ConstantBuilder cb(DynCast<Function>(c_lhs->GetParent()));
  std::vector<T> ret;
  ret.reserve(num_elements);
  switch (opcode) {
    case OpCode::ADD: {
      for (size_t i = 0; i < num_elements; ++i) {
        ret.push_back(*lhs_it++ + *rhs_it++);
      }
      c_ret = cb.CreateConstant(name, ret_type, ret.data());
      break;
    }
    case OpCode::SUB: {
      for (size_t i = 0; i < num_elements; ++i) {
        ret.push_back(*lhs_it++ - *rhs_it++);
      }
      c_ret = cb.CreateConstant(name, ret_type, ret.data());
      break;
    }
    case OpCode::MUL: {
      for (size_t i = 0; i < num_elements; ++i) {
        ret.push_back(*lhs_it++ * *rhs_it++);
      }
      c_ret = cb.CreateConstant(name, ret_type, ret.data());
      break;
    }
    case OpCode::DIV: {
      for (size_t i = 0; i < num_elements; ++i) {
        ret.push_back(*lhs_it++ / *rhs_it++);
      }
      c_ret = cb.CreateConstant(name, ret_type, ret.data());
      break;
    }
    case OpCode::CMP: {
      std::vector<int8_t> ret;
      switch (pred) {
        case KindPredicate::LT: {
          for (size_t i = 0; i < num_elements; ++i) {
            if (*lhs_it++ < *rhs_it++) {
              ret.push_back(1);
            } else {
              ret.push_back(0);
            }
          }
          c_ret = cb.CreateConstant(name, ret_type, ret.data());
          break;
        }
        case KindPredicate::EQ: {
          for (size_t i = 0; i < num_elements; ++i) {
            if (*lhs_it++ == *rhs_it++) {
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

static bool IsSameType(const Type& lhs, const Type& rhs) {
  if (lhs.GetNumOfDims() != rhs.GetNumOfDims()) {
    return false;
  }
  for (size_t d = 0; d < lhs.GetNumOfDims(); d++) {
    if (lhs.GetNumOfElementsInDim(d) != rhs.GetNumOfElementsInDim(d)) {
      return false;
    }
  }
  return true;
}

static std::pair<Def, Def> RunOnMathUnaryInstruction(Instruction* inst) {
  auto op0 = DynCast<Constant>(inst->GetOperand(0));
  Def orig_def{inst, 0};

  if (op0 == nullptr ||
      op0->GetResultType().GetDataType() != DataType::FLOAT32) {
    return {orig_def, orig_def};
  }
  auto n = op0->GetResultType().GetTotalNumOfElements();
  std::vector<float> result(n);
  std::function<float(float)> op;
  switch (inst->GetOpCode()) {
    case OpCode::SQRT:
      op = sqrtf;
      break;
    default:
      return {orig_def, orig_def};
  }

  for (int64_t i = 0; i < n; ++i) {
    result[i] = op(op0->GetDataAsFloat32(i));
  }
  ConstantBuilder cb(inst->GetParent()->GetParent());
  auto c = cb.CreateConstant(inst->GetName(), inst->GetResultType(), result);
  return {orig_def, *c};
}

static std::pair<Def, Def> RunOnMathBinaryInstruction(
    Instruction* binary_inst, bool disable_broadcasting, bool fuse_conv_bias,
    bool fuse_hardswish, bool fuse_matmul_mul, bool fuse_fully_connected) {
  Def orig_def{binary_inst, 0};
  auto op0 = binary_inst->GetOperand(0);
  auto op1 = binary_inst->GetOperand(1);
  auto opc = binary_inst->GetOpCode();
  bool has_swapped = false;
  bool commutative = opc == OpCode::ADD || opc == OpCode::MUL;

  Constant* op0_c = DynCast<Constant>(op0);
  Constant* op1_c = DynCast<Constant>(op1);

  if (op0_c != nullptr && commutative) {
    std::swap(op0, op1);
    std::swap(op0_c, op1_c);
    has_swapped = true;
  }

  // MUL(x, 1) ==> x.
  if (opc == OpCode::MUL && op1_c != nullptr) {
    if (op1_c->HasSameValueOf(1)) {
      const Type& type0 = op0.GetType();
      const Type& type1 = op1.GetType();
      if (IsSameType(type0, type1)) {
        return {orig_def, op0};
      }
    }
  }

  IRBuilder builder(binary_inst->GetParent());
  builder.SetInsertAfter(binary_inst);
  // ADD/SUB(x, 0) ==> x
  if ((opc == OpCode::ADD || opc == OpCode::SUB) && op1_c != nullptr &&
      op1_c->HasSameValueOf(0)) {
    return {orig_def, op0};
  }
  // SUB(0, x) ==> -x
  if (opc == OpCode::SUB && op0_c != nullptr && op0_c->HasSameValueOf(0)) {
    auto neg = builder.CreateNeg(binary_inst->GetName(), op1);
    return {orig_def, *neg};
  }

  ConstantBuilder cb(binary_inst->GetParent()->GetParent());
  // SUB(x, x) ==> 0
  if (opc == OpCode::SUB && op0 == op1) {
    const auto& ty = binary_inst->GetResultType();
    DefaultDataLayout dl;
    std::vector<char> data(dl.DataLayout::Bytes(ty), 0);
    auto zero = cb.CreateConstant(binary_inst->GetName(), ty, data.data());
    return {orig_def, *zero};
  }

  // fuse to fully_connected
  if (opc == OpCode::ADD && fuse_fully_connected) {
    if (IsA<MatMulInst>(op1)) {
      std::swap(op0, op1);
      std::swap(op0_c, op1_c);
      has_swapped = !has_swapped;
    }
    if (IsA<MatMulInst>(op0) && op1_c != nullptr) {
      int16_t kernel_dim =
          DynCast<MatMulInst>(op0)->GetOperand(1).GetType().GetNumOfDims();
      if ((!DynCast<MatMulInst>(op0)->GetTransposeA()) &&
          (!DynCast<MatMulInst>(op0)->GetTransposeB()) && (kernel_dim == 2)) {
        auto op_matmul0 = DynCast<MatMulInst>(op0)->GetOperand(0);
        auto op_matmul1 = DynCast<MatMulInst>(op0)->GetOperand(1);
        if (IsA<Constant>(op_matmul0)) {
          std::swap(op_matmul0, op_matmul1);
        }

        Instruction* new_inst = nullptr;

        if (IsA<Constant>(op_matmul1)) {
          TransposeInst* op_matmul1_t = builder.CreateTranspose(
              DynCast<Constant>(op_matmul1)->GetName() + "_t", {op_matmul1});

          op_matmul1_t->SetPermutation({1, 0});
          new_inst = builder.CreateGemm(binary_inst->GetName() + "_fused",
                                        op_matmul0, *op_matmul1_t, op1);
          if (new_inst != nullptr) {
            GemmInst* new_gemm = DynCast<GemmInst>(new_inst);
            new_gemm->SetTransposeB(true);
            return {orig_def, *new_gemm};
          }
        }
      }
    }
  }

  if (fuse_matmul_mul && opc == OpCode::MUL) {
    if (op0_c != nullptr) {
      std::swap(op0, op1);
      std::swap(op0_c, op1_c);
      has_swapped = !has_swapped;
    }
    if (IsA<MatMulInst>(op0) && op1_c != nullptr) {
      auto op_matmul0 = DynCast<MatMulInst>(op0)->GetOperand(0); // input
      auto op_matmul1 = DynCast<MatMulInst>(op0)->GetOperand(1); // weight
      if (IsA<Constant>(op_matmul1) &&
          DynCast<Constant>(op_matmul1)->GetResultType().GetNumOfDims() == 2 &&
          DynCast<Constant>(op_matmul1)->GetResultType().GetDataType() ==
              halo::DataType::FLOAT32) {
        MatMulInst* matmul = DynCast<MatMulInst>(op0);

        ConstantBuilder cb(matmul->GetParent()->GetParent());
        IRBuilder builder(matmul->GetParent());
        builder.SetInsertAfter(matmul);

        const auto kernel = DynCast<Constant>(op_matmul1);
        const auto& kernel_type = kernel->GetResultType();
        int32_t h = kernel_type.GetNumOfElementsInDim(0);
        int32_t w = kernel_type.GetNumOfElementsInDim(1);
        auto weight_2 = op1_c;
        std::vector<float> mul_buf(h * w);

        for (int32_t i = 0; i < w; i++) {
          for (int32_t j = 0; j < h; j++) {
            auto a = kernel->GetData<float>(j + i * w);
            auto b = weight_2->GetData<float>(i);
            auto c = a * b;
            mul_buf[i * w + j] = c;
          }
        }

        MatMulInst* new_mm = nullptr;
        auto new_kernel =
            cb.CreateConstant(kernel->GetName(), kernel_type, mul_buf.data());

        if (!DynCast<MatMulInst>(op0)->GetTransposeB()) {
          TransposeInst* new_kernel_t = builder.CreateTranspose(
              DynCast<Constant>(new_kernel)->GetName() + "_t", {*new_kernel});
          new_kernel_t->SetPermutation({1, 0});

          auto new_inst = builder.Clone(*matmul, {op_matmul0, *new_kernel_t});
          new_mm = DynCast<MatMulInst>(new_inst);
          new_mm->SetTransposeB(true);
        } else {
          auto new_inst = builder.Clone(*matmul, {op_matmul0, *new_kernel});
          new_mm = DynCast<MatMulInst>(new_inst);
        }

        return {orig_def, *new_mm};
      }
    }
  }

  // Fuse mul/add into conv.
  if ((opc == OpCode::MUL || (fuse_conv_bias && opc == OpCode::ADD)) &&
      op1_c != nullptr) {
    // check if mul can be fused with conv
    Instruction* new_inst = nullptr;
    if (IsA<Conv2DInst>(op0)) {
      new_inst = FuseToConvDeConv(DynCast<Conv2DInst>(op0), opc, op1_c);
    } else if (IsA<Conv2DTransposeInst>(op0)) {
      new_inst =
          FuseToConvDeConv(DynCast<Conv2DTransposeInst>(op0), opc, op1_c);
    }
    if (new_inst != nullptr) {
      if (fuse_conv_bias && opc == OpCode::ADD) {
        new_inst->SetName(binary_inst->GetName() + "_fused");
      }
      return {orig_def, *new_inst};
    }
  }

  if (opc == OpCode::DIV && IsA<Constant>(op1) && fuse_hardswish) {
    const Constant* c = DynCast<Constant>(op1);
    constexpr int div_6 = 6;
    constexpr int add_3 = 3;
    if (c->HasSameValueOf(div_6)) {
      auto op_mul0 = DynCast<MulInst>(op0)->GetOperand(0);
      auto op_mul1 = DynCast<MulInst>(op0)->GetOperand(1);
      auto opc_mul = DynCast<MulInst>(op0)->GetOpCode();
      if (opc_mul == OpCode::MUL) {
        auto op_relu6 = DynCast<Relu6Inst>(op_mul1)->GetOperand(0);
        auto opc_relu6 = DynCast<Relu6Inst>(op_mul1)->GetOpCode();
        if (opc_relu6 == OpCode::RELU6) {
          auto op_add0 = DynCast<AddInst>(op_relu6)->GetOperand(0);
          auto op_add1 = DynCast<AddInst>(op_relu6)->GetOperand(1);
          auto opc_add = DynCast<AddInst>(op_relu6)->GetOpCode();
          if (opc_add == OpCode::ADD && IsA<Constant>(op_add1)) {
            const Constant* c_add = DynCast<Constant>(op_add1);
            if (c_add->HasSameValueOf(add_3)) {
              Instruction* new_inst = nullptr;
              new_inst = builder.CreateHardSwish(
                  binary_inst->GetName() + "_fused", op_add0);
              return {orig_def, *new_inst};
            }
          }
        }
      }
    }
  }

  const auto& op0_type = op0.GetType();
  const auto& op1_type = op1.GetType();
  OpCode opcode = binary_inst->GetOpCode();

  // Try constant folding
  const auto& ret_type = binary_inst->GetResultType();

  KindPredicate pred = KindPredicate::INVALID;
  if (opcode == OpCode::CMP) {
    pred = DynCast<CmpInst>(binary_inst)->GetPredicator();
  }

  if (ret_type.IsValid()) {
    if (has_swapped) {
      std::swap(op0, op1);
      std::swap(op0_c, op1_c);
      has_swapped = !has_swapped;
    }
    Constant* c_ret = nullptr;
    switch (op0_type.GetDataType()) {
      case DataType::INT32: {
        c_ret = RunConstantFoldingOnMathBinary<int>(
            binary_inst->GetName() + "_folded", ret_type, op0, op1, opcode,
            pred);
        break;
      }
      case DataType::INT64: {
        c_ret = RunConstantFoldingOnMathBinary<int64_t>(
            binary_inst->GetName() + "_folded", ret_type, op0, op1, opcode,
            pred);
        break;
      }
      case DataType::FLOAT32: {
        c_ret = RunConstantFoldingOnMathBinary<float>(
            binary_inst->GetName() + "_folded", ret_type, op0, op1, opcode,
            pred);
        break;
      }
      default:
        c_ret = nullptr;
    }
    if (c_ret != nullptr) {
      return {orig_def, *c_ret};
    }
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
    auto new_add = has_swapped
                       ? builder.CreateBinary(binary_inst->GetName(), *addend,
                                              op0, opcode, pred)
                       : builder.CreateBinary(binary_inst->GetName(), op0,
                                              *addend, opcode, pred);
    new_add->GetResultsTypes()[0] = binary_inst->GetResultsTypes()[0];
    return {orig_def, *new_add};
  }

  if (op0_type.IsValid() && IsA<Constant>(op1) && IsA<TransposeInst>(op0) &&
      op1_type.BroadcastableTo(op0_type)) {
    Instruction* op0_inst = DynCast<Instruction>(op0);
    if (!IsA<Argument>(op0_inst->GetOperand(0))) {
      // Add(transpose(op0), op1) ==> transpose(add(op0, transpose'(op1))
      // if rank_1 < rank_0, we first unsqueeze(reshape) op1 to the same rank as
      // op0. Then we do the reverse transpose on op1.
      // For example: given add(To_NCHW(op0<n, h, w, c>), op1<c, 1, 1,>), we do
      //  To_NCHW(add op0<n, h, w, c>, To_NHWC(op1<1, c, 1, 1>))
      TransposeInst* orig_transpose = DynCast<TransposeInst>(op0_inst);
      IRBuilder builder(binary_inst->GetParent());
      builder.SetInsertAfter(binary_inst);
      const auto& orig_perm = orig_transpose->GetPermutation();
      Instruction* new_op1 = nullptr;
      if (auto rank_0 = op0_type.GetNumOfDims(),
          rank_1 = op1_type.GetNumOfDims();
          rank_1 < rank_0) {
        auto dims = op1_type.GetDimSizes();
        for (unsigned i = 0; i < rank_0 - rank_1; ++i) {
          dims.insert(dims.begin(), 1);
        }
        ConstantBuilder cb(binary_inst->GetParent()->GetParent());
        Constant* c_shape = cb.CreateConstant(
            op1.GetDef()->GetName() + "_shape",
            halo::Type{DataType::INT64, {static_cast<int64_t>(dims.size())}},
            dims.data());
        auto new_addend = builder.CreateReshape(op1.GetDef()->GetName() + "_r",
                                                {op1, *c_shape});
        new_addend->GetResultsTypes()[0] =
            Type{op1.GetType().GetDataType(), dims};
        op1 = *new_addend;
      }
      auto reverse_perm = orig_perm;
      for (int i = 0, e = orig_perm.size(); i < e; ++i) {
        reverse_perm[orig_perm[i]] = i;
      }
      auto new_addend =
          builder.CreateTranspose(op1.GetDef()->GetName() + "_t", {op1});
      new_addend->SetPermutation(reverse_perm);
      new_op1 = new_addend;

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

  // ReduceMean(tranpose(x, {t0, t1, t2, t3}, {a0, a1, a2...}) =>
  // ReduceMean(x, permed_axis)
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
      return RunOnMathBinaryInstruction(
          inst, opts_.disable_broadcasting, opts_.fuse_conv_bias,
          opts_.fuse_hardswish, opts_.fuse_matmul_mul,
          opts_.fuse_fully_connected);
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
    case OpCode::SQRT: {
      return RunOnMathUnaryInstruction(inst);
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
  Def orig_def{inst, 0};
  auto input = inst->GetOperand(0);
  auto input_op = IsA<Instruction>(input)
                      ? DynCast<Instruction>(input)->GetOpCode()
                      : OpCode::INVALID;
  bool is_profitable =
      input_op == OpCode::CONV2D || input_op == OpCode::CONV2DTRANSPOSE;
  if (!is_profitable || !opts_.fuse_conv_relu) {
    return SinkTranspose(
        *inst, [](IRBuilder& builder, const std::string& name, const Def& op) {
          return builder.CreateRelu6(name, op);
        });
  }

  if (input_op == OpCode::CONV2D || input_op == OpCode::CONV2DTRANSPOSE) {
    input.GetDef()->SetName(inst->GetName() + "_fused");
    return {orig_def, input};
  }

  return {orig_def, orig_def};
}

std::pair<Def, Def> InstSimplify::RunOnInstruction(ShapeInst* inst) {
  const auto& type = inst->GetOperand(0).GetType();

  Def orig_def{inst, 0};
  if (!type.IsValid() || type.IsDynamicShape() || type.IsDynamicBatch()) {
    return {orig_def, orig_def};
  }

  DataType dt = inst->GetDataType();
  ConstantBuilder cb(inst->GetParent()->GetParent());
  int64_t rank = type.GetNumOfDims();
  if (dt == DataType::INT32) {
    std::vector<int32_t> shape;
    for (int64_t i : type.GetDimSizes()) {
      shape.push_back(static_cast<int>(i));
    }
    Constant* c = cb.CreateConstant(inst->GetName(), halo::Type{dt, {rank}},
                                    shape.data());
    return {orig_def, *c};
  }
  HLCHECK(dt == DataType::INT64);
  Constant* c = cb.CreateConstant(inst->GetName(), halo::Type{dt, {rank}},
                                  type.GetDimSizes());
  return {orig_def, *c};
}

std::pair<Def, Def> InstSimplify::RunOnInstruction(SigmoidInst* inst) {
  return SinkTranspose(
      *inst, [](IRBuilder& builder, const std::string& name, const Def& op) {
        return builder.CreateSigmoid(name, op);
      });
}

std::pair<Def, Def> InstSimplify::RunOnInstruction(ReluInst* inst) {
  Def orig_def{inst, 0};
  auto input = inst->GetOperand(0);
  auto input_op = IsA<Instruction>(input)
                      ? DynCast<Instruction>(input)->GetOpCode()
                      : OpCode::INVALID;
  bool is_profitable =
      input_op == OpCode::CONV2D || input_op == OpCode::CONV2DTRANSPOSE;
  if (!is_profitable || !opts_.fuse_conv_relu) {
    return SinkTranspose(
        *inst, [](IRBuilder& builder, const std::string& name, const Def& op) {
          return builder.CreateRelu(name, op);
        });
  }

  if (input_op == OpCode::CONV2D || input_op == OpCode::CONV2DTRANSPOSE) {
    input.GetDef()->SetName(inst->GetName() + "_fused");
    return {orig_def, input};
  }
  return {orig_def, orig_def};
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
  if ((IsA<AddInst>(op_input) ||
       (IsA<MulInst>(op_input) && opts_.fuse_mul_to_conv)) &&
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

std::pair<Def, Def> InstSimplify::RunOnInstruction(NoOpInst* noop_inst) {
  return {*noop_inst,
          noop_inst->GetNumOfOperands() > 0 ? Def::GetUndefined() : *noop_inst};
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
  const auto& ret_type = inst->GetResultType();
  Def orig_def{inst, 0};

  if (!ret_type.IsValid()) {
    return {orig_def, orig_def};
  }

  auto input = inst->GetOperand(0);
  const auto& input_type = input.GetType();
  auto input_elem = input_type.GetTotalNumOfElements();
  auto ret_elem = ret_type.GetTotalNumOfElements();

  IRBuilder builder(inst->GetParent());
  builder.SetInsertAfter(inst);

  ConstantBuilder cb(inst->GetParent()->GetParent());
  if (input_elem == ret_elem) {
    Constant* c = cb.CreateConstant(
        inst->GetName() + "_expand",
        halo::Type{DataType::INT64,
                   {static_cast<int64_t>(ret_type.GetNumOfDims())}},
        ret_type.GetDimSizes().data());
    auto reshape =
        builder.CreateReshape(inst->GetName(), {inst->GetOperand(0), *c});
    return {orig_def, *reshape};
  }

  if (const Constant* src = DynCast<Constant>(inst->GetOperand(0));
      src != nullptr) {
    int result_rank = input_type.GetNumOfDims();
    int input_rank = input_type.GetNumOfDims();
    std::vector<int64_t> output_extends;
    auto src_extends = GetExtends(input_type.GetDimSizes());
    for (int i = 0, e = std::max(result_rank, input_rank); i < e; ++i) {
      int input_idx = input_rank - 1 - i;
      bool is_bs =
          input_idx < 0 || input_type.GetNumOfElementsInDim(input_idx) == 1;
      output_extends.push_back(is_bs ? 0 : src_extends[input_idx]);
    }
    std::reverse(output_extends.begin(), output_extends.end());

    DefaultDataLayout data_layout;
    size_t elem_size = data_layout.Bytes(input_type.GetDataType());
    std::vector<unsigned char> buf(ret_elem * elem_size);
    const auto& dst_extends = GetExtends(ret_type.GetDimSizes());
    for (int64_t dst_idx = 0; dst_idx < ret_elem; ++dst_idx) {
      std::vector<int64_t> dst_dims(ret_type.GetNumOfDims());
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
  auto input_op = IsA<Instruction>(input)
                      ? DynCast<Instruction>(input)->GetOpCode()
                      : OpCode::INVALID;
  bool is_profitable = input_op == OpCode::CONV2D ||
                       input_op == OpCode::CONV2DTRANSPOSE ||
                       input_op == OpCode::CONCAT;
  if (opts_.disable_conv_bn || !is_profitable || num_inputs <= 4 ||
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
  auto new_add = builder.CreateAdd(inst->GetName(), *new_mul, *new_offset);
  return {orig_def, *new_add};
}

std::pair<Def, Def> InstSimplify::RunOnInstruction(StackInst* inst) {
  Def orig_def{inst, 0};
  int num_inputs = inst->GetNumOfOperands();

  bool all_scalars = true;
  for (const auto& op : inst->GetOperands()) {
    all_scalars &= op.GetType().IsValid() && op.GetType().IsScalar();
  }
  if (all_scalars) {
    IRBuilder builder(inst->GetParent());
    builder.SetInsertAfter(inst);
    HLCHECK(inst->GetAxis() == 0);
    auto concat = builder.CreateConcat(inst->GetName(), inst->GetOperands());
    concat->SetAxis(0);
    concat->SetN(inst->GetOperands().size());
    return {orig_def, *concat};
  }
  for (int i = 0; i < num_inputs; ++i) {
    if (!IsA<Constant>(inst->GetOperand(i))) {
      return {orig_def, orig_def};
    }
  }
  const auto& input0_type = inst->GetOperand(0).GetType();
  int rank = input0_type.GetNumOfDims();
  int axis = inst->GetAxis();
  if (axis < 0) {
    axis += rank + 1;
  }
  if (axis != 0) {
    return {orig_def, orig_def};
  }
  // convert to an array of constant
  std::vector<int64_t> ret_shape = input0_type.GetDimSizes();
  ret_shape.insert(ret_shape.begin() + axis, num_inputs);
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

std::pair<Def, Def> InstSimplify::RunOnInstruction(
    SquaredDifferenceInst* inst) {
  Def orig_def{inst, 0};
  if (!opts_.convert_squared_diff) {
    return {orig_def, orig_def};
  }
  IRBuilder builder(inst->GetParent());
  builder.SetInsertAfter(inst);
  const auto& lhs = inst->GetOperand(0);
  const auto& rhs = inst->GetOperand(1);
  auto sub_inst = builder.CreateSub(inst->GetName() + "_sub", lhs, rhs);
  auto mul_inst =
      builder.CreateMul(inst->GetName() + "_square", *sub_inst, *sub_inst);
  return {orig_def, *mul_inst};
}

std::pair<Def, Def> InstSimplify::RunOnInstruction(ZExtInst* inst) {
  Def orig_def{inst, 0};
  DataType ret_dt = inst->GetDataType();
  auto op0 = inst->GetOperand(0);
  const auto& op0_type = op0.GetType();
  DataType src_dt = op0_type.GetDataType();

  if (!op0_type.IsValid() || !IsA<Constant>(op0)) {
    return {orig_def, orig_def};
  }

  HLCHECK(halo::Type::IsIntegerType(src_dt) &&
          halo::Type::IsIntegerType(ret_dt));

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

std::pair<Def, Def> InstSimplify::RunOnInstruction(GatherElementsInst* inst) {
  HLCHECK(inst->GetNumOfOperands() == 2);
  auto input_op = inst->GetOperand(0);
  auto idx_op = inst->GetOperand(1);
  auto const& input_type = input_op.GetType();
  auto const& idx_type = idx_op.GetType();
  Def orig_def{inst, 0};

  if (!input_type.IsValid() || !idx_type.IsValid()) {
    return {orig_def, orig_def};
  }

  const auto& input_shape = input_type.GetDimSizes();
  auto idx_shape = idx_type.GetDimSizes();

  int axis = inst->GetAxis();
  int rank = input_type.GetNumOfDims();
  axis = axis < 0 ? rank + axis : axis;
  inst->SetAxis(axis);

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
    ConstantBuilder cb(inst->GetParent()->GetParent());
    std::vector<int64_t> new_dims{idx_shape[axis]};
    Constant* c = cb.CreateConstant(
        inst->GetName() + "_shape",
        halo::Type{DataType::INT64, {static_cast<int64_t>(new_dims.size())}},
        new_dims.data());
    IRBuilder builder(inst->GetParent());
    builder.SetInsertAfter(inst);
    auto reshape =
        builder.CreateReshape(inst->GetName() + "_reshape", idx_op, *c);
    auto new_inst = builder.CreateGather(inst->GetName(), {input_op, *reshape});
    new_inst->SetAxis(axis);
    return {orig_def, *new_inst};
  }
  return {orig_def, orig_def};
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

static Constant* FoldConcatConstant(ConcatInst* inst) {
  if (!inst->GetResultType().IsValid()) {
    return nullptr;
  }

  int num_inputs = inst->GetN();

  std::vector<Constant*> input_cs(num_inputs);

  for (int i = 0; i < num_inputs; ++i) {
    Constant* opi = DynCast<Constant>(inst->GetOperand(i));

    if (opi == nullptr) {
      return nullptr;
    }

    input_cs[i] = opi;
  }

  const Type& op0_type = input_cs.at(0)->GetResultType();
  int dim_num = static_cast<int>(op0_type.GetNumOfDims());

  int64_t axis = inst->GetAxis();
  if (axis < 0) {
    axis += op0_type.GetNumOfDims();
  }

  int64_t total_num_elems = inst->GetResultType().GetTotalNumOfElements();
  int element_byte_size = input_cs.at(0)->GetElementSizeInBytes();
  int64_t total_bytes = total_num_elems * element_byte_size;

  int base_size = element_byte_size;

  auto data = std::vector<uint8_t>(total_bytes);
  uint8_t* dst = data.data();

  // Scalars and 1D-tensors need special treatement
  if (op0_type.IsScalar() || dim_num == 1) {
    for (Constant* c : input_cs) {
      size_t num_bytes =
          c->GetResultType().GetTotalNumOfElements() * element_byte_size;
      const uint8_t* src =
          static_cast<const uint8_t*>(c->GetRawDataPtr()); // NOLINT
      std::copy_n(src, num_bytes, dst);
      dst += num_bytes; // NOLINT
    }

    std::vector<int64_t> dims{total_num_elems};

    ConstantBuilder cb(inst->GetParent()->GetParent());
    std::string name = inst->GetName() + "_folding";
    return cb.CreateConstant(name, inst->GetResultType(), data.data());
  }

  // Normal tensors
  for (int64_t i = axis + 1; i < dim_num; ++i) {
    base_size *= op0_type.GetNumOfElementsInDim(i);
  }

  int total_num_elements_along_axis = 0;
  std::vector<int> chunk_sizes;
  std::vector<const uint8_t*> src_ptrs;

  for (Constant* opi : input_cs) {
    int num_elements_along_axis =
        opi->GetResultType().GetNumOfElementsInDim(axis);

    chunk_sizes.push_back(num_elements_along_axis * base_size);

    src_ptrs.push_back(
        static_cast<const uint8_t*>(opi->GetRawDataPtr())); // NOLINT

    total_num_elements_along_axis += num_elements_along_axis;
  }

  int group_num = 1;
  for (int i = 0; i < axis; ++i) {
    group_num *= op0_type.GetNumOfElementsInDim(i);
  }

  for (int i = 0; i < group_num; ++i) {
    for (int j = 0; j < num_inputs; ++j) {
      std::copy_n(src_ptrs[j], chunk_sizes[j], dst);
      src_ptrs[j] += chunk_sizes[j]; // NOLINT
      dst += chunk_sizes[j];         // NOLINT
    }
  }

  std::vector<int64_t> dims(op0_type.GetDimSizes());
  dims[axis] = total_num_elements_along_axis;

  ConstantBuilder cb(inst->GetParent()->GetParent());
  std::string name = inst->GetName() + "_folding";
  Type new_type(op0_type.GetDataType(), dims);

  return cb.CreateConstant(name, new_type, data.data());
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

  // Skip empty inputs.
  std::vector<Def> operands;
  for (const auto& op : inst->GetOperands()) {
    if (!op.GetType().IsValid() || op.GetType().GetTotalNumOfElements() != 0) {
      operands.push_back(op);
    }
  }
  if (operands.size() < inst->GetNumOfOperands()) {
    IRBuilder builder(inst->GetParent());
    builder.SetInsertAfter(inst);

    auto new_concat = builder.Clone(*inst, operands);
    return {orig_def, *new_concat};
  }

  for (size_t i = 0; i < inst->GetNumOfOperands(); ++i) {
    if (!IsA<Constant>(inst->GetOperand(i).GetOwner())) {
      return {orig_def, orig_def};
    }
  }

  if (!inst->GetResultsTypes()[0].IsValid()) {
    return {orig_def, orig_def};
  }

  if (auto new_inst = FoldConcatConstant(inst)) {
    return {orig_def, *new_inst};
  }

  return {orig_def, orig_def};
}

class CopyTileData {
 public:
  CopyTileData(const void* src, void* dst, int64_t elem_byte_size,
               const std::vector<int64_t>& dims,
               const std::vector<int64_t>& multiplies)
      : src_(static_cast<const uint8_t*>(src)),
        dst_(static_cast<uint8_t*>(dst)),
        elem_byte_size_(elem_byte_size),
        dims_(dims),
        multiplies_(multiplies) {}

  void Run() { RunImpl(0); }

 private:
  void RunImpl(size_t index) {
    if (index + 1 == dims_.size()) {
      int64_t total_bytes = dims_[index] * elem_byte_size_;
      int factor = multiplies_[index];

      for (int i = 0; i < factor; ++i, dst_ += total_bytes) { // NOLINT
        std::copy_n(src_, total_bytes, dst_);
      }

      src_ += total_bytes; // NOLINT

    } else {
      int num_elems = dims_[index];

      // The dst_ will be increased in the following recursions.
      const uint8_t* initial_dst = dst_;

      for (int i = 0; i < num_elems; ++i) {
        RunImpl(index + 1);
      }

      int factor = multiplies_[index];

      int64_t total_bytes = dst_ - initial_dst;

      for (int i = 1; i < factor; ++i, dst_ += total_bytes) { // NOLINT
        std::copy_n(initial_dst, total_bytes, dst_);
      }
    }
  }

 private:
  const uint8_t* src_;
  uint8_t* dst_;
  int64_t elem_byte_size_;
  const std::vector<int64_t>& dims_;
  const std::vector<int64_t>& multiplies_;
};

enum TileArgIndex { TILE_ARG_INPUT_IDX = 0, TILE_ARG_MULTIPLES_IDX = 1 };

std::pair<Def, Def> InstSimplify::RunOnInstruction(TileInst* inst) {
  Def orig_def{inst, 0};

  if (!inst->GetResultType().IsValid()) {
    return {orig_def, orig_def};
  }

  auto c_input = DynCast<Constant>(inst->GetOperand(TILE_ARG_INPUT_IDX));
  if (c_input == nullptr || !c_input->GetResultType().IsValid()) {
    return {orig_def, orig_def};
  }

  auto c_multiplies =
      DynCast<Constant>(inst->GetOperand(TILE_ARG_MULTIPLES_IDX));
  if (c_multiplies == nullptr || !c_multiplies->GetResultType().IsValid()) {
    return {orig_def, orig_def};
  }

  const halo::Type& input_type = c_input->GetResultType();
  const std::vector<int64_t>& input_dims = input_type.GetDimSizes();

  std::vector<int64_t> multiplies(input_dims.size());
  std::vector<int64_t> output_dims(input_dims);

  for (size_t i = 0; i < output_dims.size(); ++i) {
    int64_t m = c_multiplies->GetDataAsInt64(i);
    multiplies[i] = m;
    output_dims[i] *= m;
  }

  int64_t total_num_elems = std::accumulate(
      output_dims.begin(), output_dims.end(), 1, std::multiplies<>{});

  size_t total_byte_sizes = total_num_elems * c_input->GetElementSizeInBytes();
  auto buffer = std::make_unique<uint8_t[]>(total_byte_sizes); // NOLINT

  CopyTileData worker(c_input->GetRawDataPtr(), buffer.get(),
                      c_input->GetElementSizeInBytes(), input_dims, multiplies);

  worker.Run();

  ConstantBuilder cb(inst->GetParent()->GetParent());
  std::string name = inst->GetName() + "_folding";
  halo::Type output_type(input_type.GetDataType(), output_dims);
  Constant* c_ret = cb.CreateConstant(name, output_type, buffer.get());
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
  const auto& input_type = input.GetType();
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

  if (opts_.remove_input_transpose && input.GetUses().size() == 1) {
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
      new_perm[i] = perm0[perm[i]];
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
  if (!opts_.remove_output_transpose) {
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

template <typename T>
static Constant* GetSlicedConstantDataRankOne(Instruction* inst,
                                              Constant* input, int idx,
                                              int len) {
  const auto& dt = inst->GetResultType();
  std::vector<T> data(dt.GetTotalNumOfElements());

  HLCHECK(static_cast<size_t>(len) == data.size());
  for (int i = 0; i < len; ++i) {
    data[i] = input->GetData<T>(idx + i);
  }
  ConstantBuilder cb(inst->GetParent()->GetParent());
  return cb.CreateConstant(inst->GetName(), dt, data.data());
}

struct SingleDim {
  int64_t First;
  int64_t Last;
  int64_t DimSize;
};

class CopySliceData {
 public:
  CopySliceData(const void* src, void* dst, int element_byte_size,
                std::vector<SingleDim> ranges)
      : src_(static_cast<const uint8_t*>(src)),
        dst_(static_cast<uint8_t*>(dst)),
        group_offset_(0),
        dst_offset_(0),
        ranges_(std::move(ranges)),
        scales_(ranges_.size() - 1),
        element_size_(element_byte_size),
        chunk_offset_(ranges_.back().First),
        chunk_size_((ranges_.back().Last - ranges_.back().First) *
                    element_size_) {
    for (int i = static_cast<int>(ranges_.size()) - 1, acc = 1; i > 0; --i) {
      acc *= ranges_[i].DimSize;
      scales_[i - 1] = acc;
    }
  }

  void Run() { RunImpl(0); }

 private:
  // TODO(lingqing.zz): refactor the recursion into an iteration.
  void RunImpl(int dim) {
    if (dim + 1 == static_cast<int>(ranges_.size())) {
      int64_t src_offset = (group_offset_ + chunk_offset_) * element_size_;
      std::copy_n(src_ + src_offset, chunk_size_, // NOLINT
                  dst_ + dst_offset_);            // NOLINT
      dst_offset_ += chunk_size_;
    } else {
      int first = ranges_[dim].First;
      int last = ranges_[dim].Last;

      for (int i = first; i < last; ++i) {
        int64_t delta = i * scales_[dim];
        group_offset_ += delta;
        RunImpl(dim + 1);
        group_offset_ -= delta;
      }
    }
  }

 private:
  const uint8_t* src_;
  uint8_t* dst_;
  int64_t group_offset_;
  int64_t dst_offset_;
  std::vector<SingleDim> ranges_;
  std::vector<int> scales_;
  int64_t element_size_;
  int64_t chunk_offset_;
  int64_t chunk_size_;
};

enum SliceArgIndex {
  SLICE_ARG_INPUT_IDX = 0,
  SLICE_ARG_STARTS_IDX = 1,
  SLICE_ARG_SIZES_IDX = 2,
  SLICE_ARG_STEPS_IDX = 3,
  SLICE_ARG_AXES_IDX = 4
};

struct SliceParameters {
  Constant* Input = nullptr;
  Constant* Starts = nullptr;
  Constant* Sizes = nullptr;
  Constant* Steps = nullptr;
  Constant* Axes = nullptr;
};

static bool IsFoldableSlice(SliceInst* inst, SliceParameters* params) {
  Constant* c_input = DynCast<Constant>(inst->GetOperand(SLICE_ARG_INPUT_IDX));
  if (c_input == nullptr || !c_input->GetResultType().IsValid()) {
    return false;
  }

  Constant* starts_c =
      DynCast<Constant>(inst->GetOperand(SLICE_ARG_STARTS_IDX));
  if (starts_c == nullptr || !starts_c->GetResultType().IsValid()) {
    return false;
  }

  Constant* slice_sizes_c =
      DynCast<Constant>(inst->GetOperand(SLICE_ARG_SIZES_IDX));
  if (slice_sizes_c == nullptr || !slice_sizes_c->GetResultType().IsValid()) {
    return false;
  }

  if (inst->GetNumOfOperands() <= SLICE_ARG_STEPS_IDX) {
    return false;
  }

  Constant* steps_c = DynCast<Constant>(inst->GetOperand(SLICE_ARG_STEPS_IDX));
  if (steps_c == nullptr || !steps_c->GetResultType().IsValid()) {
    return false;
  }

  // TODO(lingqing.zz): Relax the restriction.
  if (!steps_c->HasSameValueOf(1)) {
    return false;
  }

  if (inst->GetNumOfOperands() <= SLICE_ARG_AXES_IDX) {
    return false;
  }

  Constant* axes_c = DynCast<Constant>(inst->GetOperand(SLICE_ARG_AXES_IDX));
  if (axes_c == nullptr || !axes_c->GetResultType().IsValid()) {
    return false;
  }

  params->Input = c_input;
  params->Starts = starts_c;
  params->Sizes = slice_sizes_c;
  params->Steps = steps_c;
  params->Axes = axes_c;

  return true;
}

static Constant* FoldSliceInst(SliceInst* inst) {
  SliceParameters params;

  if (!IsFoldableSlice(inst, &params)) {
    return nullptr;
  }

  const Type& axes_type = params.Axes->GetResultType();
  int64_t axes_num = axes_type.GetTotalNumOfElements();

  const Type& input_type = params.Input->GetResultType();
  std::vector<int64_t> dims(input_type.GetDimSizes());
  std::vector<SingleDim> ranges(input_type.GetNumOfDims());

  for (size_t i = 0; i < input_type.GetNumOfDims(); ++i) {
    ranges[i] = {0, dims[i], dims[i]};
  }

  for (int64_t i = 0; i < axes_num; ++i) {
    int64_t axis = params.Axes->GetDataAsInt64(i);
    int64_t start = params.Starts->GetDataAsInt64(i);
    int64_t size = params.Sizes->GetDataAsInt64(i);

    dims[axis] = size;
    ranges[axis].First = start;
    ranges[axis].Last = start + size;
  }

  int total_elements =
      std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<>{});
  std::size_t bytes_size =
      total_elements * params.Input->GetElementSizeInBytes();

  auto data = std::make_unique<uint8_t[]>(bytes_size); // NOLINT

  CopySliceData worker(params.Input->GetRawDataPtr(), data.get(),
                       params.Input->GetElementSizeInBytes(),
                       std::move(ranges));
  worker.Run();

  ConstantBuilder builder(inst->GetParent()->GetParent());
  std::string name = inst->GetName() + "_folded";
  Type slice_type(input_type.GetDataType(), dims);
  return builder.CreateConstant(name, slice_type, data.get());
}

std::pair<Def, Def> InstSimplify::RunOnInstruction(SliceInst* inst) {
  Def orig_def{inst, 0};
  auto op_len = inst->GetOperand(2);
  const auto& dst_type = inst->GetResultsTypes()[0];
  if (dst_type.IsValid() && !dst_type.IsStaticShape()) {
    IRBuilder builder(inst->GetParent());
    builder.SetInsertAfter(inst);
    ConstantBuilder cb(inst->GetParent()->GetParent());

    auto input = inst->GetOperand(0);
    ShapeInst* shape_input =
        builder.CreateShape(inst->GetName() + "_inputshape", input);

    std::vector<Def> concat_operands;
    const halo::Type size_i_type{DataType::INT64, {1}};
    int64_t size_len = 1;
    Constant* c_len = cb.CreateConstant(inst->GetName() + "_size_len",
                                        size_i_type, &size_len);
    int dim = dst_type.GetNumOfDims();
    for (int i = 0; i != dim; ++i) {
      int64_t dim_i = dst_type.GetNumOfElementsInDim(i);
      if (dim_i == -1) {
        int64_t start = i;
        Constant* shape_slice_start = cb.CreateConstant(
            inst->GetName() + "_size_" + std::to_string(i) + "_start",
            size_i_type, &start);
        auto slice_i =
            builder.CreateSlice(inst->GetName() + "_size_" + std::to_string(i),
                                {*shape_input, *shape_slice_start, *c_len});
        concat_operands.push_back(*slice_i);
      } else {
        Constant* c_i =
            cb.CreateConstant(inst->GetName() + "_size_" + std::to_string(i),
                              size_i_type, &dim_i);
        concat_operands.push_back(*c_i);
      }
    }
    auto dynamic_size = builder.CreateConcat(inst->GetName() + "_dynamic_size",
                                             concat_operands);
    auto new_slice = builder.CreateSliceDynamic(
        inst->GetName(),
        {inst->GetOperand(0), inst->GetOperand(1), *dynamic_size});
    return {orig_def, *new_slice};
  }
  if (dst_type.IsValid() && IsA<Constant>(op_len)) {
    Constant* c_size = DynCast<Constant>(op_len);
    int dim = op_len.GetType().GetTotalNumOfElements();
    std::vector<int> size_adj(dim);
    bool new_size = false;
    for (int i = 0; i != dim; ++i) {
      int64_t size_i = c_size->GetDataAsInt64(i);
      int64_t s = dst_type.GetNumOfElementsInDim(i);
      if (size_i == -1 && s != -1) {
        size_adj[i] = s;
        new_size = true;
      } else {
        size_adj[i] = size_i;
      }
    }
    if (new_size) {
      ConstantBuilder cb(inst->GetParent()->GetParent());
      Constant* c_new_size = cb.CreateConstant(
          op_len.GetOwner()->GetName() + "_adj",
          halo::Type{DataType::INT32, op_len.GetType().GetDimSizes()},
          size_adj.data());
      IRBuilder builder(inst->GetParent());
      builder.SetInsertAfter(inst);
      SliceInst* new_inst = builder.CreateSlice(
          inst->GetName(),
          {inst->GetOperand(0), inst->GetOperand(1), *c_new_size});
      new_inst->GetResultsTypes()[0] = dst_type;
      return {orig_def, *new_inst};
    }
  }
  const auto& op0 = inst->GetOperand(0);
  const auto& op_start = inst->GetOperand(1);

  bool has_constant_steps =
      (inst->GetNumOfOperands() < 4 || IsA<Constant>(inst->GetOperand(3)));
  has_constant_steps &=
      (inst->GetNumOfOperands() <= 4 || IsA<Constant>(inst->GetOperand(4)));

  bool has_constant_axes =
      (inst->GetNumOfOperands() <= 4 || IsA<Constant>(inst->GetOperand(4)));

  if (IsA<Constant>(op0) && IsA<Constant>(op_start) && IsA<Constant>(op_len) &&
      inst->GetResultType().IsValid() && has_constant_steps &&
      has_constant_axes) {
    Constant* input = DynCast<Constant>(op0);
    const auto& dt = inst->GetResultType();
    auto starts = DynCast<Constant>(op_start);
    auto lens = DynCast<Constant>(op_len);
    auto rank = op0.GetType().GetNumOfDims();
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
    bool all_steps_are_one = has_constant_steps;
    if (rank == 1 && all_steps_are_one) {
      auto idx = starts->GetDataAsInt64(0);
      auto len = lens->GetDataAsInt64(0);
      Constant* c = nullptr;
      switch (dt.GetDataType()) {
        case DataType::INT64:
          c = GetSlicedConstantDataRankOne<int64_t>(inst, input, idx, len);
          break;
        case DataType::INT32:
          c = GetSlicedConstantDataRankOne<int>(inst, input, idx, len);
          break;
        case DataType::FLOAT32:
          c = GetSlicedConstantDataRankOne<float>(inst, input, idx, len);
          break;
        default:
          break;
      }
      if (c != nullptr) {
        return {orig_def, *c};
      }
    } else if (auto new_inst = FoldSliceInst(inst)) {
      return {orig_def, *new_inst};
    }
  }

  // If all data to be sliced is constant, convert the result to contant.
  const Constant* c_start = DynCast<Constant>(inst->GetOperand(1));
  const Constant* c_len = DynCast<Constant>(inst->GetOperand(2));
  const auto& ret_type = inst->GetResultType();
  if (c_start != nullptr && c_len != nullptr && ret_type.IsValid() &&
      ret_type.GetNumOfDims() <= 1) {
    // so far only support simple case (input is 1-D).
    auto from = c_start->GetDataAsInt64(0);
    auto len = c_len->GetDataAsInt64(0);
    bool is_constant = true;
    std::vector<int64_t> data(len);
    for (int64_t i = 0; i < len && is_constant; ++i) {
      const auto& c = GetAvailIntegerResult(op0, from + i);
      is_constant &= c.first;
      data[i] = c.second;
    }
    if (is_constant) {
      ConstantBuilder cb(inst->GetParent()->GetParent());
      auto c = cb.CreateConstant(inst->GetName(), inst->GetResultType(),
                                 data.data());
      return {orig_def, *c};
    }
  }
  return {orig_def, orig_def};
}

template <typename T>
static Constant* GetSelectedConstant(Instruction* inst, const Constant* cond,
                                     const Constant* tv, const Constant* fv) {
  const auto& ret_type = inst->GetResultType();
  size_t num_elements = ret_type.GetTotalNumOfElements();
  std::string name = inst->GetName() + "_folded";
  ConstantBuilder cb(inst->GetParent()->GetParent());

  std::vector<T> ret;
  for (size_t i = 0; i < num_elements; ++i) {
    if (cond->GetData<bool>(i)) {
      ret.push_back(tv->GetData<T>(i));
    } else {
      ret.push_back(fv->GetData<T>(i));
    }
  }
  auto c_ret = cb.CreateConstant(name, ret_type, ret.data());
  return c_ret;
}

std::pair<Def, Def> InstSimplify::RunOnInstruction(SelectInst* inst) {
  Def orig_def{inst, 0};
  auto cond = DynCast<Constant>(inst->GetOperand(0));
  const auto& ret_type = inst->GetResultType();
  if (cond == nullptr || !ret_type.IsValid()) {
    return {orig_def, orig_def};
  }
  const auto& lhs = inst->GetOperand(1);
  const auto& rhs = inst->GetOperand(2);
  // If no broadcasting, we can simply return the true/false value.
  if (lhs.GetType() == ret_type && cond->HasSameValueOf(1)) {
    return {orig_def, lhs};
  }
  if (rhs.GetType() == ret_type && cond->HasSameValueOf(0)) {
    return {orig_def, rhs};
  }

  auto tv = DynCast<Constant>(lhs);
  auto fv = DynCast<Constant>(rhs);
  if (tv != nullptr && fv != nullptr) {
    Constant* c_ret = nullptr;
    switch (ret_type.GetDataType()) {
      case DataType::INT32: {
        c_ret = GetSelectedConstant<int32_t>(inst, cond, tv, fv);
        break;
      }
      case DataType::INT64: {
        c_ret = GetSelectedConstant<int64_t>(inst, cond, tv, fv);
        break;
      }
      case DataType::FLOAT32: {
        c_ret = GetSelectedConstant<float>(inst, cond, tv, fv);
        break;
      }
      default:
        break;
    }
    if (c_ret != nullptr) {
      return {orig_def, *c_ret};
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
      if (IsA<Argument>(op_reshape)) {
        Argument* arg = DynCast<Argument>(op_reshape);
        if (arg->GetNumberOfUses() == 1 && op_reshape.GetType().IsValid()) {
          arg->SetType(halo::Type{DataType::FLOAT32,
                                  op_reshape.GetType().GetDimSizes()});
          return {orig_def, *reshape_inst};
        }
      }
    }
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
  // work around on cxx target when backend doesn't support onehot.
  if (!opts_.simplify_for_preprocess) {
    return {orig_def, orig_def};
  }
  auto on_value = inst->GetOperand(2);
  const auto& dst_type = inst->GetResultsTypes()[0];
  if (!IsA<Constant>(on_value) ||
      (inst->GetNumOfOperands() == 4 && !IsA<Constant>(inst->GetOperand(3))) ||
      !dst_type.IsValid()) {
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
  }
  /* else if (IsA<Argument>(op0) && op0.GetType().IsValid() &&
             DynCast<Argument>(op0)->GetNumberOfUses() == 1) {
    Argument* arg = DynCast<Argument>(op0);
    arg->SetType(
        halo::Type{on_value.GetType().GetDataType(), dst_type.GetDimSizes()});
    return {orig_def, *arg};
  }*/
  return {orig_def, orig_def};
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

static bool FixUpLSTM(LSTMInst* inst) {
  bool changed = false;

  // Drop the last few all-zero operands
  while (true) {
    size_t num_ops = inst->GetNumOfOperands();
    if (num_ops <= LSTM_ARG_INITIAL_H_IDX) {
      break;
    }

    HLCHECK(num_ops > 0);
    size_t op_idx = num_ops - 1;

    Constant* op = DynCast<Constant>(inst->GetOperand(op_idx));
    if (nullptr == op || !op->IsFullOfZeros()) {
      break;
    }

    inst->ResetOperand(op_idx);
    inst->GetOperands().pop_back();
    changed = true;
  }

  if (LSTM_ARG_SEQUENCE_LENGTH_IDX + 1 == inst->GetNumOfOperands()) {
    // Now the sequence_lens is the last argument

    const Type& type_x = inst->GetOperand(LSTM_ARG_X_IDX).GetType();

    int64_t seq_length = type_x.GetNumOfElementsInDim(0);
    int64_t batch_size = type_x.GetNumOfElementsInDim(1);

    if (LSTM_LAYOUT_NORMAL != inst->GetLayout()) {
      std::swap(seq_length, batch_size);
    }

    Constant* op =
        DynCast<Constant>(inst->GetOperand(LSTM_ARG_SEQUENCE_LENGTH_IDX));

    if (nullptr != op) {
      // If the sequence_lens was filled with default values, drop it.
      if (op->HasSameValueOf(seq_length)) {
        inst->ResetOperand(LSTM_ARG_SEQUENCE_LENGTH_IDX);
        inst->GetOperands().pop_back();
        changed = true;
      }
    }
  }

  return changed;
}

std::pair<Def, Def> InstSimplify::RunOnInstruction(MatMulInst* inst) {
  // for matmul(x, transpose(y), false, false) ==> matmul(x,
  // transpose2(transpose(y)), false, true). Nested transposes will be
  // combined or cancelled.
  Def orig_def{inst, 0};
  const auto& op1 = inst->GetOperand(1);
  if (const TransposeInst* trans = DynCast<TransposeInst>(op1);
      trans != nullptr && !inst->GetTransposeB() &&
      trans->GetNumberOfUses() == 1) {
    auto perm = trans->GetPermutation();
    HLCHECK(perm.size() >= 2);
    std::iota(perm.begin(), perm.end(), 0);
    std::swap(perm[perm.size() - 1], perm[perm.size() - 2]);
    IRBuilder builder(inst->GetParent());
    builder.SetInsertBefore(inst);
    auto new_transpose_inst =
        builder.CreateTranspose(trans->GetName() + "_t", {op1});
    new_transpose_inst->SetPermutation(perm);
    auto new_matmul = DynCast<MatMulInst>(
        builder.Clone(*inst, {inst->GetOperand(0), *new_transpose_inst}));
    new_matmul->SetTransposeB(true);
    return {orig_def, *new_matmul};
  }
  return {orig_def, orig_def};
}

std::pair<Def, Def> InstSimplify::RunOnInstruction(MeanInst* inst) {
  Def orig_def{inst, 0};
  auto operands = inst->GetOperands();
  if (operands.size() == 1) {
    return {orig_def, operands[0]};
  }
  return {orig_def, orig_def};
}

std::pair<Def, Def> InstSimplify::RunOnInstruction(UniqueInst* inst) {
  Def orig_def{inst, 1};
  if (!opts_.simplify_for_preprocess) {
    return {orig_def, orig_def};
  }
  const auto& result_type0 = inst->GetResultsTypes()[0];
  const auto& result_type1 = inst->GetResultsTypes()[1];
  auto num_uses = inst->GetResultsUses()[0].size();
  if (!result_type0.IsValid() || !result_type1.IsValid() || num_uses != 0) {
    return {orig_def, orig_def};
  }

  auto bitcast_inst = DynCast<BitcastInst>(inst->GetOperand(0));
  if (bitcast_inst != nullptr && bitcast_inst->GetNumberOfUses() == 1) {
    auto stack_inst = DynCast<StackInst>(bitcast_inst->GetOperand(0));
    if (stack_inst != nullptr && stack_inst->GetNumberOfUses() == 1) {
      bool check_all = true;
      for (size_t i = 0; i < stack_inst->GetNumOfOperands(); ++i) {
        const auto& op_i = stack_inst->GetOperand(i);
        if (!IsA<Argument>(op_i) || !op_i.GetUses().HasOneUse()) {
          check_all = false;
          break;
        }
      }
      if (check_all) {
        ArgumentBuilder arg_builder(inst->GetParent()->GetParent());
        auto arg = arg_builder.CreateArgument(inst->GetName() + "_preprocess",
                                              result_type1);
        return {orig_def, *arg};
      }
    }
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

    if (auto lstm = DynCast<LSTMInst>(inst)) {
      changed |= FixUpLSTM(lstm);
    }
  }
  return changed;
}

} // end namespace halo
