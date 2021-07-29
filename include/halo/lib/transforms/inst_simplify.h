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

#include "halo/lib/ir/all_instructions.h"
#include "halo/lib/pass/pass.h"

namespace halo {

/// This pass simplififies instructions to reduce computation strength.
class InstSimplify final : public BasicBlockPass {
 public:
  InstSimplify() : InstSimplify(false, false, false, false, false, false) {}
  InstSimplify(bool simplify_for_preprocess, bool disable_broadcasting,
               bool remove_input_transpose, bool remove_output_transpose,
               bool disable_conv_bn, bool fuse_conv_bias)
      : BasicBlockPass("Instruction Simplification"),
        simplify_for_preprocess_(simplify_for_preprocess),
        disable_broadcasting_(disable_broadcasting),
        remove_input_transpose_(remove_input_transpose),
        remove_output_transpose_(remove_output_transpose),
        disable_conv_bn_(disable_conv_bn),
        fuse_conv_bias_(fuse_conv_bias) {}

  bool RunOnBasicBlock(BasicBlock* bb) override;

 private:
  // TODO(unknown): Tablegen.
  std::pair<Def, Def> RunOnInstruction(Instruction* inst);
  std::pair<Def, Def> RunOnInstruction(TransposeInst* inst);
  std::pair<Def, Def> RunOnInstruction(ReturnInst* inst);
  std::pair<Def, Def> RunOnInstruction(BatchNormInst* inst);
  static std::pair<Def, Def> RunOnInstruction(CeilInst* inst);
  static std::pair<Def, Def> RunOnInstruction(ConcatInst* inst);
  static std::pair<Def, Def> RunOnInstruction(Conv2DInst* inst);
  static std::pair<Def, Def> RunOnInstruction(ExpandDimsInst* inst);
  static std::pair<Def, Def> RunOnInstruction(GatherInst* inst);
  static std::pair<Def, Def> RunOnInstruction(NoOpInst* inst);
  static std::pair<Def, Def> RunOnInstruction(PadInst* inst);
  static std::pair<Def, Def> RunOnInstruction(RangeInst* inst);
  static std::pair<Def, Def> RunOnInstruction(RandomUniformInst* inst);
  static std::pair<Def, Def> RunOnInstruction(ReshapeInst* inst);
  static std::pair<Def, Def> RunOnInstruction(ReluInst* inst);
  static std::pair<Def, Def> RunOnInstruction(Relu6Inst* inst);
  static std::pair<Def, Def> RunOnInstruction(PReluInst* inst);
  static std::pair<Def, Def> RunOnInstruction(TileInst* inst);
  static std::pair<Def, Def> RunOnInstruction(LeakyReluInst* inst);
  static std::pair<Def, Def> RunOnInstruction(ResizeInst* inst);
  static std::pair<Def, Def> RunOnInstruction(SetDiff1DInst* inst);
  static std::pair<Def, Def> RunOnInstruction(SigmoidInst* inst);
  static std::pair<Def, Def> RunOnInstruction(SItoFPInst* inst);
  static std::pair<Def, Def> RunOnInstruction(FPtoSIInst* inst);
  static std::pair<Def, Def> RunOnInstruction(SliceInst* inst);
  static std::pair<Def, Def> RunOnInstruction(StackInst* inst);
  static std::pair<Def, Def> RunOnInstruction(ZExtInst* inst);

  std::pair<Def, Def> RunOnInstruction(OneHotInst* inst);

  bool simplify_for_preprocess_;
  bool disable_broadcasting_;
  bool remove_input_transpose_;
  bool remove_output_transpose_;
  bool disable_conv_bn_;
  bool fuse_conv_bias_;
};

} // end namespace halo.

#endif // HALO_LIB_TRANSFORMS_INST_SIMPLIFY_H_
