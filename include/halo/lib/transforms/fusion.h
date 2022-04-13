//===- fusion.h -----------------------------------------------------------===//
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

#ifndef HALO_LIB_TRANSFORMS_FUSION_H_
#define HALO_LIB_TRANSFORMS_FUSION_H_

#include "halo/lib/ir/all_instructions.h"
#include "halo/lib/pass/pass.h"

namespace halo {

/// This pass simplififies instructions to reduce computation strength.
struct FusionOptions {
#define HALO_FUSION_OPTIONS
#include "halo/lib/ir/fusion.cc.inc"
#undef HALO_FUSION_OPTIONS
  bool FuseLayerNorm = true;
  bool FuseGelu = true;
  bool FuseMHA = true;
};

class Fusion final : public BasicBlockPass {
 public:
  Fusion(const FusionOptions& opts)
      : BasicBlockPass("Instruction Fusion"), opts_(opts) {}

  bool RunOnBasicBlock(BasicBlock* bb) override;

  struct FusionOptions opts_;

 private:
  std::pair<Def, Def> RunOnInstruction(Instruction* inst);
};

} // end namespace halo.

#endif // HALO_LIB_TRANSFORMS_FUSION_H_
