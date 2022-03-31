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

// ming
static std::pair<bool, float> IsScalar(const Constant* constant) {
  bool is_scalar = constant != nullptr && constant->GetResultType().IsScalar();
  return is_scalar ? std::make_pair(true, constant->GetDataAsFloat32(0))
                   : std::make_pair(false, NAN);
}

static bool IsScalar(const Constant* constant, float x) {
  auto result = IsScalar(constant);
  return result.first && result.second == x;
}

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
        input_(Def::GetUndefined()),
        mask_(Def::GetUndefined()),
        query_t_(Def::GetUndefined()),
        query_bias_(Def::GetUndefined()),
        key_t_(Def::GetUndefined()),
        key_bias_(Def::GetUndefined()),
        value_t_(Def::GetUndefined()),
        value_bias_(Def::GetUndefined()) {
    matched_ = MatchMHA(inst);
  }

  bool Matched() const {
    return matched_ && input_.GetType().GetTotalNumOfElements() ==
                           batch_ * seq_len_ * heads_ * hidden_size_;
  }

  Def GetFusedMHA() const {
    if (!Matched()) {
      return Def::GetUndefined();
    }
    IRBuilder builder(output_inst_->GetParent());
    builder.SetInsertAfter(output_inst_);
    auto ret = builder.CreateCustom(output_inst_->GetName() + "_mha",
                                    GetOperands(), 1, "MHA");
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
                 : Def::GetUndefined();
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

  static inline bool IsValidTranspose(const TransposeInst* transpose) {
    if (transpose == nullptr || transpose->GetNumOfOperands() != 1 ||
        !transpose->GetResultType().IsValid()) {
      return false;
    }
    const std::vector<int> expected_perm{0, 2, 1, 3};
    const auto& perm = transpose->GetPermutation();
    return std::equal(perm.begin(), perm.end(), expected_perm.begin());
  }

  bool MatchQKV(const Def& op, Def* weight, Def* bias) {
    const TransposeInst* transpose = DynCast<TransposeInst>(op);
    if (!IsValidTranspose(transpose)) {
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
      input_ = Def::GetUndefined();
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
    *bias = gemm->GetNumOfOperands() > 2 ? gemm->GetOperand(2)
                                         : Def::GetUndefined();
    return true;
  }

  bool MatchQKBase(const BatchMatMulInst* matmul) {
    if (matmul == nullptr || matmul->GetTransposeA() ||
        !matmul->GetTransposeB()) {
      return false;
    }
    return MatchQKV(matmul->GetOperand(0), &query_t_, &query_bias_) &&
           MatchQKV(matmul->GetOperand(1), &key_t_, &key_bias_);
  }

  bool MatchQKBase(const MulInst* mul) {
    if (mul == nullptr || heads_ <= 0) {
      return false;
    }
    auto mul_lhs = mul->GetOperand(0);
    auto mul_rhs = mul->GetOperand(1);
    float scale = 1.0F / sqrtf(static_cast<float>(hidden_size_));
    if (!IsScalar(DynCast<Constant>(mul_rhs), scale)) {
      std::swap(mul_lhs, mul_rhs);
    }

    return IsScalar(DynCast<Constant>(mul_rhs), scale) &&
           MatchQKBase(DynCast<BatchMatMulInst>(mul_lhs));
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
    auto matmul = DynCast<BatchMatMulInst>(inst->GetOperand(0));
    if (matmul == nullptr) {
      return false;
    }
    const Type& dt = matmul->GetResultType();
    if (!dt.IsValid() || matmul->GetTransposeA() || matmul->GetTransposeB() ||
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
           MatchQKV(rhs, &value_t_, &value_bias_);
  }
}; // namespace halo
// ming

// mry
// need you
class LayerNormMatcher {
 public:
  explicit LayerNormMatcher(const Instruction* inst)
      : output_inst_(inst),
        matched_(false),
        epsilon_(0.0F),
        input_(Def::GetUndefined()),
        gamma_(Def::GetUndefined()),
        beta_(Def::GetUndefined()),
        gamma_div_std_(Def::GetUndefined()) {
    matched_ = MatchLayerNorm(inst);
  }

  bool Matched() const { return matched_; }
  Def GetFusedLayerNorm() const {
    if (!Matched()) {
      return Def::GetUndefined();
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
    // Check if op1 == mean(op0)
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
    // check for (x - mean(x))^2
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
    // Match for variance = mean(squared_difference(op0, mean(op0)))

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
    if (!IsA<RsqrtInst>(op)) {
      return false;
    }
    auto rsqrt_inst = DynCast<RsqrtInst>(op);
    const auto& op0 = rsqrt_inst->GetOperand(0);
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

  static bool MatchMean(const Def& op0, const Def& op1) {
    // Check if op1 == mean(op0)
    auto mean_inst = DynCast<ReduceMeanInst>(op1);
    return (mean_inst != nullptr && mean_inst->GetOperand(0) == op0);
  }
  bool MatchGammaMeanDivStd(const Def& op) {
    if (!IsA<MulInst>(op)) {
      return false;
    }
    auto mul_inst = DynCast<MulInst>(op);
    const auto& lhs = mul_inst->GetOperand(0);
    const auto& rhs = mul_inst->GetOperand(1);
    if (lhs == gamma_div_std_) {
      return MatchMean(input_, rhs);
    }
    if (rhs == gamma_div_std_) {
      return MatchMean(input_, lhs);
    }
    return false;
  }

  bool MatchBetaSubGammaMeanDivStd(const Def& op) {
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
      // match  (x * gamma / sqrt(var + epsilon)) + (beta - mean(x)*
      // gamma/sqrt(var + epsilon))
      auto lhs = inst->GetOperand(0);
      auto rhs = inst->GetOperand(1);
      return MatchLayerNormPatterns(lhs, rhs);
    }
    return false;
  }
}; // namespace halo

// mry

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
    } else {
      MultiHeadAttentionMatcher mha_matcher(inst);
      if (mha_matcher.Matched()) {
        changed |= true;
        inst->ReplaceAllUsesWith(0, mha_matcher.GetFusedMHA());
        continue;
      }
      LayerNormMatcher ln_matcher(inst);
      if (ln_matcher.Matched()) {
        changed |= true;
        inst->ReplaceAllUsesWith(0, ln_matcher.GetFusedLayerNorm());
      }
    }
  }
  return changed;
}

} // end namespace halo