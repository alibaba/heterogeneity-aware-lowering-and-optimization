//===- liveness_analyzer.cc -----------------------------------------------===//
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

#include "halo/lib/mm/memory_analyzer.h"

#include "halo/lib/framework/data_layout.h"

namespace halo {

void MemoryAnalyzer::RunOnFunction(const Function& func) {
  const DataLayout& dl = ctx_.GetDefaultDataLayout();
  for (auto& constant : func.Constants()) {
    use_cnts_[constant->GetId()] = constant->GetNumberOfUses();
    const Type& ty = constant->GetResultType();
    weights_ += dl.Bytes(ty);
  }
  for (auto& bb : func.BasicBlocks()) {
    for (auto& inst : bb->Instructions()) {
      use_cnts_[inst->GetId()] = inst->GetNumberOfUses();
      for (auto& ty : inst->GetResultsTypes()) {
        non_weights_ += dl.Bytes(ty);
      }
    }
  }
}

std::vector<Def> MemoryAnalyzer::Executed(const Instruction* inst) {
  const DataLayout& dl = ctx_.GetDefaultDataLayout();
  std::vector<Def> dead;
  for (auto& ty : inst->GetResultsTypes()) {
    curr_non_weights_ += dl.Bytes(ty);
  }
  for (const auto& op : inst->GetOperands()) {
    if (!IsA<Instruction>(op)) {
      continue;
    }
    auto id = op.GetOwner()->GetId();
    --curr_use_cnts_[id];
    if (curr_use_cnts_[id] == 0) {
      dead.push_back(op);
      curr_non_weights_ -= dl.Bytes(op.GetType());
    }
  }
  peak_ = std::max(peak_, curr_non_weights_);
  return dead;
}

MemoryAnalyzer::MemoryAnalyzer(const Module& m)
    : module_(m),
      ctx_(m.GetGlobalContext()),
      weights_(0),
      non_weights_(0),
      curr_non_weights_(0),
      peak_(0) {
  for (auto& func : m) {
    RunOnFunction(*func);
  }
  Reset();
}

} // namespace halo