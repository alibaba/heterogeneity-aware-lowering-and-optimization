//===- inst_simplify.h ----------------------------------------------------===//
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

#ifndef HALO_LIB_TRANSFORMS_INST_SIMPLIFY_H_
#define HALO_LIB_TRANSFORMS_INST_SIMPLIFY_H_

#include "halo/halo.h"
#include "halo/lib/ir/all_instructions.h"
#include "halo/lib/pass/pass.h"

namespace halo {

/// This pass simplififies instructions to reduce computation strength.
class InstSimplify final : public BasicBlockPass {
 public:
  InstSimplify() : BasicBlockPass("Instruction Simplification") {}
  InstSimplify(const CXXCodeGenOpts& opts)
      : BasicBlockPass("Instruction Simplification"), opts_(opts) {}

  bool RunOnBasicBlock(BasicBlock* bb) override;

 private:
  // TODO(unknown): Tablegen.
  std::pair<Def, Def> RunOnInstruction(Instruction* inst);
  std::pair<Def, Def> RunOnInstruction(TransposeInst* inst);
  std::pair<Def, Def> RunOnInstruction(ReturnInst* inst);
  std::pair<Def, Def> RunOnInstruction(BatchNormInst* inst);
  static std::pair<Def, Def> RunOnInstruction(CeilInst* inst);
  static std::pair<Def, Def> RunOnInstruction(ConcatInst* inst);
  std::pair<Def, Def> RunOnInstruction(Conv2DInst* inst);
  static std::pair<Def, Def> RunOnInstruction(ExpandDimsInst* inst);
  static std::pair<Def, Def> RunOnInstruction(GatherInst* inst);
  static std::pair<Def, Def> RunOnInstruction(GatherElementsInst* inst);
  static std::pair<Def, Def> RunOnInstruction(MatMulInst* inst);
  static std::pair<Def, Def> RunOnInstruction(MeanInst* inst);
  static std::pair<Def, Def> RunOnInstruction(NoOpInst* inst);
  static std::pair<Def, Def> RunOnInstruction(PadInst* inst);
  static std::pair<Def, Def> RunOnInstruction(RangeInst* inst);
  static std::pair<Def, Def> RunOnInstruction(RandomUniformInst* inst);
  static std::pair<Def, Def> RunOnInstruction(ReshapeInst* inst);
  std::pair<Def, Def> RunOnInstruction(ReluInst* inst);
  std::pair<Def, Def> RunOnInstruction(Relu6Inst* inst);
  static std::pair<Def, Def> RunOnInstruction(PReluInst* inst);
  static std::pair<Def, Def> RunOnInstruction(TileInst* inst);
  static std::pair<Def, Def> RunOnInstruction(LeakyReluInst* inst);
  static std::pair<Def, Def> RunOnInstruction(ResizeInst* inst);
  static std::pair<Def, Def> RunOnInstruction(SelectInst* inst);
  static std::pair<Def, Def> RunOnInstruction(SetDiff1DInst* inst);
  static std::pair<Def, Def> RunOnInstruction(ShapeInst* inst);
  static std::pair<Def, Def> RunOnInstruction(SigmoidInst* inst);
  static std::pair<Def, Def> RunOnInstruction(SItoFPInst* inst);
  static std::pair<Def, Def> RunOnInstruction(FPtoSIInst* inst);
  static std::pair<Def, Def> RunOnInstruction(SliceInst* inst);
  static std::pair<Def, Def> RunOnInstruction(StackInst* inst);
  static std::pair<Def, Def> RunOnInstruction(ZExtInst* inst);

  std::pair<Def, Def> RunOnInstruction(OneHotInst* inst);
  std::pair<Def, Def> RunOnInstruction(UniqueInst* inst);
  std::pair<Def, Def> RunOnInstruction(SquaredDifferenceInst* inst);

  CXXCodeGenOpts opts_;
};

} // end namespace halo.

#endif // HALO_LIB_TRANSFORMS_INST_SIMPLIFY_H_
