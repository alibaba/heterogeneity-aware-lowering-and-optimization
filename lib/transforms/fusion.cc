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
    Def op0 = Def::GetUndefined();
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
}; // namespace halo

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
    } else if (opts_.FuseLayerNorm) {
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