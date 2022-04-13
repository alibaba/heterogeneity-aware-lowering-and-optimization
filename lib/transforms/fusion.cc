//===- fusion.cc ----------------------------------------------------------===//
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

#include "halo/lib/transforms/fusion.h"

#include <cmath>

#include "halo/api/halo_data.h"
#include "halo/lib/framework/common.h"
#include "halo/lib/framework/data_layout.h"
#include "halo/lib/framework/global_context.h"
#include "halo/lib/ir/ir_builder.h"

namespace halo {

static const Def& Undefined = Def::GetUndefined();

static std::pair<bool, float> IsScalar(const Constant* constant) {
  bool is_scalar = constant != nullptr && constant->GetResultType().IsScalar();
  return is_scalar ? std::make_pair(true, constant->GetDataAsFloat32(0))
                   : std::make_pair(false, NAN);
}

static bool IsScalar(const Constant* constant, float x) {
  auto result = IsScalar(constant);
  return result.first && result.second == x;
}

static bool IsScalar(const Def& def, float x) {
  return IsScalar(DynCast<Constant>(def), x);
}

static const Def& MatchOneOperand(const Def& target, const Def& lhs,
                                  const Def& rhs) {
  return lhs == target ? rhs : (rhs == target ? lhs : Undefined);
}

static Def MatchOneOperand(float target, const Def& lhs, const Def& rhs) {
  return IsScalar(lhs, target) ? rhs
                               : (IsScalar(rhs, target) ? lhs : Undefined);
}

static Def MatchOneOperand(const Def& target, const Instruction* inst) {
  return (inst == nullptr || inst->GetNumOfOperands() != 2)
             ? Undefined
             : MatchOneOperand(target, inst->GetOperand(0),
                               inst->GetOperand(1));
}

static Def MatchOneOperand(float target, const Instruction* inst) {
  return (inst == nullptr || inst->GetNumOfOperands() != 2)
             ? Undefined
             : MatchOneOperand(target, inst->GetOperand(0),
                               inst->GetOperand(1));
}

class LayerNormMatcher {
 public:
  explicit LayerNormMatcher(const Instruction* inst)
      : output_inst_(inst),
        matched_(false),
        epsilon_(0.0F),
        input_(Undefined),
        gamma_(Undefined),
        beta_(Undefined),
        gamma_div_std_(Undefined) {
    matched_ = MatchLayerNorm(inst);
  }

  bool Matched() const { return matched_; }
  Def GetFusedLayerNorm() const {
    if (!Matched()) {
      return Undefined;
    }
    IRBuilder builder(output_inst_->GetParent());
    builder.SetInsertAfter(output_inst_);
    auto ret = builder.CreateLayerNorm(output_inst_->GetName() + "_layernorm",
                                       GetOperands());
    ret->SetAxis(dims_);
    ret->SetEpsilon(epsilon_);
    return *ret;
  }
  std::vector<Def> GetOperands() const {
    std::vector<Def> ops{input_, gamma_, beta_};
    return ops;
  }

  float GetEpsilon() const { return epsilon_; }
  std::vector<int> GetDims() const { return dims_; }

 private:
  const Instruction* output_inst_;
  bool matched_;
  float epsilon_;
  std::vector<int> dims_;
  Def input_;
  Def gamma_;
  Def beta_;
  Def gamma_div_std_;

  bool MatchMeanDiff(const Def& lhs, const Def& rhs) {
    // Check if rhs == mean(lhs)
    // Get input_ and check dims_
    if (IsA<ReduceMeanInst>(rhs)) {
      auto mean_inst = DynCast<ReduceMeanInst>(rhs);
      if (mean_inst->GetOperand(0) == lhs) {
        const auto& axis = mean_inst->GetAxis();
        if (axis.size() == dims_.size() &&
            std::equal(axis.begin(), axis.end(), dims_.begin())) {
          input_ = lhs;
          return true;
        }
      }
    }
    return false;
  }

  bool MatchSquaredDiff(const Def& op) {
    // Match (x - mean(x))^2
    // SquaredDiff or (Sub + Mul)
    if (IsA<SquaredDifferenceInst>(op)) {
      auto inst = DynCast<SquaredDifferenceInst>(op);
      const auto& lhs = inst->GetOperand(0);
      const auto& rhs = inst->GetOperand(1);
      return MatchMeanDiff(lhs, rhs);
    }
    if (IsA<MulInst>(op)) {
      auto mul_inst = DynCast<MulInst>(op);
      auto sub_op = mul_inst->GetOperand(0);
      if (IsA<SubInst>(sub_op) && sub_op == mul_inst->GetOperand(1)) {
        auto inst = DynCast<SubInst>(sub_op);
        const auto& lhs = inst->GetOperand(0);
        const auto& rhs = inst->GetOperand(1);
        return MatchMeanDiff(lhs, rhs);
      }
    }
    return false;
  }

  bool MatchVariance(const Def& op) {
    // Match mean(squared_difference(op0, mean(op0)))
    // Get dims_
    if (!IsA<ReduceMeanInst>(op)) {
      return false;
    }
    auto reducemean_inst = DynCast<ReduceMeanInst>(op);
    dims_ = reducemean_inst->GetAxis();
    // check for square_diff(x, mean(x))
    auto op0 = reducemean_inst->GetOperand(0);
    return MatchSquaredDiff(op0);
  }

  bool MatchVarRsqrt(const Def& op) {
    // Match  1 / sqrt(var + epsilon)
    // Rsqrt or (Sqrt + Rcp)
    // Get epsilon_
    Def op0 = Undefined;
    if (IsA<RsqrtInst>(op)) {
      auto rsqrt_inst = DynCast<RsqrtInst>(op);
      op0 = rsqrt_inst->GetOperand(0);
    } else if (IsA<RcpInst>(op)) {
      auto rcp_inst = DynCast<RcpInst>(op);
      auto sqrt_op = rcp_inst->GetOperand(0);
      if (IsA<SqrtInst>(sqrt_op)) {
        auto sqrt_inst = DynCast<SqrtInst>(sqrt_op);
        op0 = sqrt_inst->GetOperand(0);
      }
    } else {
      return false;
    }
    if (IsA<AddInst>(op0)) {
      auto add_inst = DynCast<AddInst>(op0);
      auto get_epsilon = [](const Def& def) {
        if (const Constant* c = DynCast<Constant>(def)) {
          if (c->GetResultType().GetTotalNumOfElements() == 1) {
            return std::make_pair(true, c->GetDataAsFloat32(0));
          }
        }
        return std::make_pair(false, 0.0f);
      };
      const auto& lhs = add_inst->GetOperand(0);
      const auto& rhs = add_inst->GetOperand(1);
      auto ret = get_epsilon(lhs);
      if (ret.first) {
        epsilon_ = ret.second;
        return MatchVariance(rhs);
      }
      ret = get_epsilon(rhs);
      if (ret.first) {
        epsilon_ = ret.second;
        return MatchVariance(lhs);
      }
    }
    // No epsilon.
    return MatchVariance(op0);
  }

  bool MatchGammaDivStd(const Def& op) {
    // Match  gamma / sqrt(var + epsilon)
    // Get gamma_ and gamma_div_std_

    if (!IsA<MulInst>(op)) {
      return false;
    }
    auto mul_inst = DynCast<MulInst>(op);
    const auto& lhs = mul_inst->GetOperand(0);
    const auto& rhs = mul_inst->GetOperand(1);
    if (IsA<Constant>(lhs)) {
      if (MatchVarRsqrt(rhs)) {
        gamma_ = lhs;
        gamma_div_std_ = op;
        return true;
      }
    }
    if (IsA<Constant>(rhs)) {
      if (MatchVarRsqrt(lhs)) {
        gamma_ = rhs;
        gamma_div_std_ = op;
        return true;
      }
    }
    return false;
  }

  bool MatchGammaXDivStd(const Def& op) {
    // Match  x * gamma / sqrt(var + epsilon)
    if (!IsA<MulInst>(op)) {
      return false;
    }
    auto mul_inst = DynCast<MulInst>(op);
    const auto& lhs = mul_inst->GetOperand(0);
    const auto& rhs = mul_inst->GetOperand(1);
    if (MatchGammaDivStd(lhs)) {
      if (input_ == rhs) {
        return true;
      }
    }
    if (MatchGammaDivStd(rhs)) {
      if (input_ == lhs) {
        return true;
      }
    }
    return false;
  }

  bool MatchGammaMeanDivStd(const Def& op) {
    // Match mean(x) * gamma/sqrt(var + epsilon)

    if (!IsA<MulInst>(op)) {
      return false;
    }
    auto mul_inst = DynCast<MulInst>(op);
    const auto& lhs = mul_inst->GetOperand(0);
    const auto& rhs = mul_inst->GetOperand(1);
    if (lhs == gamma_div_std_) {
      return MatchMeanDiff(input_, rhs);
    }
    if (rhs == gamma_div_std_) {
      return MatchMeanDiff(input_, lhs);
    }
    return false;
  }

  bool MatchBetaSubGammaMeanDivStd(const Def& op) {
    // Match (beta - mean(x) * gamma/sqrt(var + epsilon))
    // Get beta_
    if (!IsA<SubInst>(op)) {
      return false;
    }
    auto sub_inst = DynCast<SubInst>(op);
    const auto& lhs = sub_inst->GetOperand(0);
    const auto& rhs = sub_inst->GetOperand(1);
    if (IsA<Constant>(lhs)) {
      if (MatchGammaMeanDivStd(rhs)) {
        beta_ = lhs;
        return true;
      }
    }
    return false;
  }

  bool MatchLayerNormPatterns(const Def& lhs, const Def& rhs) {
    bool matchpattern =
        (MatchGammaXDivStd(lhs) && MatchBetaSubGammaMeanDivStd(rhs)) ||
        (MatchGammaXDivStd(rhs) && MatchBetaSubGammaMeanDivStd(lhs));
    return matchpattern;
  }

  bool MatchLayerNorm(const Instruction* inst) {
    if (inst->GetOpCode() == OpCode::ADD) {
      // Match  (x * gamma / sqrt(var + epsilon)) + (beta - mean(x)*
      // gamma/sqrt(var + epsilon))
      auto lhs = inst->GetOperand(0);
      auto rhs = inst->GetOperand(1);
      return MatchLayerNormPatterns(lhs, rhs);
    }
    return false;
  }
};

class MultiHeadAttentionMatcher {
 public:
  explicit MultiHeadAttentionMatcher(const Instruction* inst)
      : output_inst_(inst),
        matched_(false),
        batch_(0),
        heads_(0),
        seq_len_(0),
        hidden_size_(0),
        has_masking_(false),
        mask_value_(NAN),
        input_(Undefined),
        mask_(Undefined),
        query_t_(Undefined),
        query_bias_(Undefined),
        key_t_(Undefined),
        key_bias_(Undefined),
        value_t_(Undefined),
        value_bias_(Undefined) {
    matched_ = MatchMHA(inst);
  }

  bool Matched() const {
    return matched_ && input_.GetType().GetTotalNumOfElements() ==
                           batch_ * seq_len_ * heads_ * hidden_size_;
  }

  Def GetFusedMHA() const {
    if (!Matched()) {
      return Undefined;
    }
    IRBuilder builder(output_inst_->GetParent());
    builder.SetInsertAfter(output_inst_);
    auto ret = builder.CreateMultiHeadAttention(output_inst_->GetName(),
                                                GetOperands());
    ret->GetResultsTypes()[0] = output_inst_->GetResultType();
    return *ret;
  }

  std::vector<Def> GetOperands() const {
    std::vector<Def> ops{input_, mask_,     query_t_, query_bias_,
                         key_t_, key_bias_, value_t_, value_bias_};

    if (IsA<ReshapeInst>(input_)) {
      ops[0] = DynCast<ReshapeInst>(input_)->GetOperand(0);
    }
    return ops;
  }

  int GetBatch() const { return batch_; }
  int GetHeads() const { return heads_; }
  int GetHiddenSize() const { return hidden_size_; }
  bool HasMasking() const { return has_masking_; }
  float GetMaskingValue() const { return mask_value_; }

 private:
  static inline constexpr int Dim = 4;
  const Instruction* output_inst_;
  bool matched_;
  int batch_;
  int heads_;
  int seq_len_;
  int hidden_size_;
  bool has_masking_;
  float mask_value_;
  Def input_;
  Def mask_;
  Def query_t_;
  Def query_bias_;
  Def key_t_;
  Def key_bias_;
  Def value_t_;
  Def value_bias_;

  bool MatchMasking(const MulInst* mul) {
    if (mul == nullptr) {
      return false;
    }
    // check for (1 - mask) * value
    auto is_one_minus_mask = [](const Def& op) {
      const Instruction* sub = DynCast<SubInst>(op);
      return sub != nullptr &&
                     IsScalar(DynCast<Constant>(sub->GetOperand(0)), 1.0f)
                 ? sub->GetOperand(1)
                 : Undefined;
    };

    auto c = IsScalar(DynCast<Constant>(mul->GetOperand(0)));
    if (c.first) {
      mask_value_ = c.second;
      mask_ = is_one_minus_mask(mul->GetOperand(1));
      return true;
    }
    c = IsScalar(DynCast<Constant>(mul->GetOperand(1)));
    if (c.first) {
      mask_value_ = c.second;
      mask_ = is_one_minus_mask(mul->GetOperand(0));
      return true;
    }
    return false;
  }

  static bool IsValidTranspose(const TransposeInst* transpose,
                               bool last_two_dims_transposed = false) {
    if (transpose == nullptr || transpose->GetNumOfOperands() != 1 ||
        !transpose->GetResultType().IsValid()) {
      return false;
    }
    const std::vector<int> expected_perm{0, 2, 1, 3};
    const std::vector<int> expected_perm_t{0, 2, 3, 1};
    const auto& perm = transpose->GetPermutation();
    return std::equal(perm.begin(), perm.end(),
                      last_two_dims_transposed ? expected_perm_t.begin()
                                               : expected_perm.begin());
  }

  bool MatchQKV(const Def& op, bool transposed, Def* weight, Def* bias) {
    const TransposeInst* transpose = DynCast<TransposeInst>(op);
    if (!IsValidTranspose(transpose, transposed)) {
      return false;
    }
    const ReshapeInst* reshape = DynCast<ReshapeInst>(transpose->GetOperand(0));
    if (reshape == nullptr || !reshape->GetResultType().IsValid()) {
      return false;
    }
    const GemmInst* gemm = DynCast<GemmInst>(reshape->GetOperand(0));
    if (gemm == nullptr || gemm->GetTransposeA() || !gemm->GetTransposeB() ||
        gemm->GetAlpha() != 1.0F || gemm->GetBeta() != 1.0F) {
      return false;
    }
    if (!input_.IsNull() && input_ != gemm->GetOperand(0)) {
      input_ = Undefined;
      return false;
    }
    if (input_.IsNull()) {
      input_ = gemm->GetOperand(0);
    }
    if (!IsA<Constant>(gemm->GetOperand(1)) ||
        (gemm->GetNumOfOperands() > 2 && !IsA<Constant>(gemm->GetOperand(2)))) {
      return false;
    }
    *weight = gemm->GetOperand(1);
    *bias = gemm->GetNumOfOperands() > 2 ? gemm->GetOperand(2) : Undefined;
    return true;
  }

  bool MatchQKBase(const MatMulInst* matmul) {
    if (matmul == nullptr || matmul->GetTransposeA() ||
        !matmul->GetTransposeB()) {
      return false;
    }
    return MatchQKV(matmul->GetOperand(0), false, &query_t_, &query_bias_) &&
           MatchQKV(matmul->GetOperand(1), false, &key_t_, &key_bias_);
  }

  bool MatchQKBase(const MulInst* mul) {
    if (mul == nullptr || heads_ <= 0) {
      return false;
    }
    float scale = 1.0F / sqrtf(static_cast<float>(hidden_size_));
    const auto& mul_lhs = MatchOneOperand(scale, mul);
    return MatchQKBase(DynCast<MatMulInst>(mul_lhs));
  }

  bool MatchQKScores(const SoftmaxInst* inst) {
    if (inst == nullptr ||
        !(inst->GetAxis() == -1 || inst->GetAxis() == Dim - 1)) {
      return false;
    }
    auto input = inst->GetOperand(0);
    if (const AddInst* add = DynCast<AddInst>(input); add != nullptr) {
      has_masking_ = true;
      bool matched = MatchQKBase(DynCast<MulInst>(add->GetOperand(0)));
      matched &= MatchMasking(DynCast<MulInst>(add->GetOperand(1)));
      if (!matched) {
        matched = MatchQKBase(DynCast<MulInst>(add->GetOperand(1)));
        matched &= MatchMasking(DynCast<MulInst>(add->GetOperand(0)));
      }
      return matched;
    }
    return MatchQKBase(DynCast<MulInst>(input));
  }

  bool MatchMHA(const Instruction* inst) {
    auto transpose = DynCast<TransposeInst>(inst);
    if (!IsValidTranspose(transpose)) {
      return false;
    }
    auto matmul = DynCast<MatMulInst>(inst->GetOperand(0));
    if (matmul == nullptr) {
      return false;
    }
    const Type& dt = matmul->GetResultType();
    if (!dt.IsValid() || matmul->GetTransposeA() ||
        matmul->GetNumOfOperands() != 2 || dt.GetNumOfDims() != Dim) {
      return false;
    }
    auto lhs = matmul->GetOperand(0);
    auto rhs = matmul->GetOperand(1);
    batch_ = dt.GetNumOfElementsInDim(0);
    heads_ = dt.GetNumOfElementsInDim(1);
    seq_len_ = dt.GetNumOfElementsInDim(2);
    hidden_size_ = dt.GetNumOfElementsInDim(3);
    return MatchQKScores(DynCast<SoftmaxInst>(lhs)) &&
           MatchQKV(rhs, matmul->GetTransposeB(), &value_t_, &value_bias_);
  }
};

class GeluMatcher {
 public:
  explicit GeluMatcher(const Instruction* inst)
      : matched_(false),
        approximated_(false),
        output_inst_(inst),
        input_(Undefined) {
    matched_ = Match();
  }
  bool Matched() const { return matched_ && !input_.IsNull(); }
  Def GetFusedGelu() const {
    if (!Matched()) {
      return Undefined;
    }
    IRBuilder builder(output_inst_->GetParent());
    builder.SetInsertAfter(output_inst_);
    auto ret = builder.CreateGelu(output_inst_->GetName(), {input_});
    ret->GetResultsTypes()[0] = output_inst_->GetResultType();
    ret->SetUseApproximation(approximated_);
    return *ret;
  }

 private:
  bool matched_;
  bool approximated_;
  const Instruction* output_inst_;
  Def input_;

  bool Match(const ErfInst* erf) {
    if (erf == nullptr) {
      return false;
    }
    if (const DivInst* div = DynCast<DivInst>(erf->GetOperand(0));
        div != nullptr) {
      if (div->GetOperand(0) != this->input_) {
        return false;
      }
      const auto& denom = div->GetOperand(1);
      if (const SqrtInst* sqrt = DynCast<SqrtInst>(denom); sqrt != nullptr) {
        constexpr float two = 2.0F;
        return IsScalar(sqrt->GetOperand(0), two);
      }
      constexpr float sqrt_2 = 1.4142135623730951F;
      return IsScalar(denom, sqrt_2);
    }
    if (const MulInst* mul = DynCast<MulInst>(erf->GetOperand(0));
        mul != nullptr) {
      constexpr float rsqrt_2 = 0.7071067811865475F;
      return MatchOneOperand(rsqrt_2, mul) == this->input_;
    }
    return false;
  }

  bool Match(const TanhInst* tanh) {
    // match for tanh(sqrt(2/PI) * (x + 0.044715 * x^3))
    if (tanh == nullptr) {
      return false;
    }
    const MulInst* mul = DynCast<MulInst>(tanh->GetOperand(0));
    constexpr float sqrt_2_pi = 0.7978845608028654F;
    const Def& mul_rhs = MatchOneOperand(sqrt_2_pi, mul);
    if (const AddInst* add = DynCast<AddInst>(mul_rhs); add != nullptr) {
      // Check for (x + 0.044715 * x^3).
      approximated_ = true;
      const auto& add_rhs = MatchOneOperand(input_, add);
      if (const MulInst* mul = DynCast<MulInst>(add_rhs); mul != nullptr) {
        constexpr float c = 0.044715F;
        const auto& pow = MatchOneOperand(c, mul);
        if (const PowInst* p = DynCast<PowInst>(pow); p != nullptr) {
          constexpr float exp = 3.0F;
          return p->GetOperand(0) == input_ && IsScalar(p->GetOperand(1), exp);
        }
      }
    }
    return false;
  }

  bool Match(const AddInst* add) {
    const auto& add_rhs = MatchOneOperand(1.0F, add);
    // match for (1+ erf(x / sqrt(2)))
    if (Match(DynCast<ErfInst>(add_rhs))) {
      return true;
    }
    return Match(DynCast<TanhInst>(add_rhs));
  }

  bool Match(std::array<Def, 3> ops) {
    // Find out 0.5
    bool found = false;
    constexpr float half = 0.5F;
    for (auto& op : ops) {
      if (IsScalar(op, half)) {
        std::swap(op, ops[2]);
        found = true;
        break;
      }
    }
    if (!found) {
      return false;
    }
    if (IsA<AddInst>(ops[0])) {
      std::swap(ops[0], ops[1]);
    }
    input_ = ops[0];
    return Match(DynCast<AddInst>(ops[1]));
  }

  bool Match() {
    // 0.5 * x * (1+ erf(x / sqrt(2)))
    auto mul = DynCast<MulInst>(output_inst_);
    if (mul == nullptr) {
      return false;
    }
    if (auto mul2 = DynCast<MulInst>(mul->GetOperand(0)); mul2 != nullptr) {
      return Match(
          {mul->GetOperand(1), mul2->GetOperand(0), mul2->GetOperand(1)});
    }
    if (auto mul2 = DynCast<MulInst>(mul->GetOperand(1)); mul2 != nullptr) {
      return Match(
          {mul->GetOperand(0), mul2->GetOperand(0), mul2->GetOperand(1)});
    }
    return false;
  }
};

static bool ValidateOpSizeAndCode(const Instruction* inst, size_t op_num,
                                  OpCode op) {
  return inst->GetNumOfOperands() == op_num && inst->GetOpCode() == op;
}

#define HALO_FUSION_MATCHERS
#include "halo/lib/ir/fusion.cc.inc"
#undef HALO_FUSION_MATCHERS

bool Fusion::RunOnBasicBlock(BasicBlock* bb) {
  bool changed = false;
  IRBuilder builder(bb);

  for (auto& inst_t : *bb) {
    Instruction* inst = inst_t.get();
    if (inst->GetNumberOfUses() == 0) {
      continue;
    }
    std::pair<Def, Def> ret{Def{inst, 0}, Def{inst, 0}};

#define HALO_FUSION_CALLS
#include "halo/lib/ir/fusion.cc.inc"
#undef HALO_FUSION_CALLS

    if (ret.first != ret.second) {
      changed |= true;
      if (ret.second.GetOwner() != nullptr) {
        // Replace all uses
        inst->ReplaceAllUsesWith(ret.first.GetIdx(), ret.second);
      }
    }
    if (opts_.FuseLayerNorm) {
      LayerNormMatcher ln_matcher(inst);
      if (ln_matcher.Matched()) {
        changed |= true;
        inst->ReplaceAllUsesWith(0, ln_matcher.GetFusedLayerNorm());
        continue;
      }
    }
    if (opts_.FuseMHA) {
      if (MultiHeadAttentionMatcher matcher(inst); matcher.Matched()) {
        changed |= true;
        inst->ReplaceAllUsesWith(0, matcher.GetFusedMHA());
        continue;
      }
    }
    if (opts_.FuseGelu) {
      if (GeluMatcher matcher(inst); matcher.Matched()) {
        changed |= true;
        inst->ReplaceAllUsesWith(0, matcher.GetFusedGelu());
        continue;
      }
    }
  }
  return changed;
}

} // end namespace halo
