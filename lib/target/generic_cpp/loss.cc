//===- loss.cc ------------------------------------------------------------===//
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

#include "halo/lib/framework/common.h"
#include "halo/lib/ir/loss_instructions.h"
#include "halo/lib/target/generic_cxx/generic_cxx_codegen.h"
#include "halo/lib/transforms/type_legalizer.h"

namespace halo {

void GenericCXXCodeGen::RunOnInstruction(NegativeLogLikelihoodLossInst* inst) {
  static const std::unordered_map<ReductionMode, std::string> mode_strs{
      {ReductionMode::None, "ODLA_REDUCE_NONE"},
      {ReductionMode::MEAN, "ODLA_REDUCE_MEAN"},
      {ReductionMode::SUM, "ODLA_REDUCE_SUM"}};

  const Def& x = inst->GetOperand(0);
  const Def& t = inst->GetOperand(1);
  const Def& w =
      inst->GetNumOfOperands() > 2 ? inst->GetOperand(2) : Def::GetUndefined();

  CXXValue op_x = ir_mapping_[x];
  CXXValue op_t = ir_mapping_[t];
  CXXValue op_w = ir_mapping_[w];
  CXXValue ret(inst->GetName(), op_x.type);

  auto mode_it = mode_strs.find(inst->GetReduction());
  const std::string& mode =
      mode_it == mode_strs.end() ? "ODLA_REDUCE_NONE" : mode_it->second;
  EmitODLACall(ret, "odla_NegativeLogLikeliHoodLoss", op_x, op_t,
               inst->GetIgnored(), mode, op_w,
               EmitShape(inst->GetResultType()));

  ir_mapping_[*inst] = ret;
}

} // namespace halo
