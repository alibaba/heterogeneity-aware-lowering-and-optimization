//===- controlflow_instructions.td ---------------------------*- tblgen -*-===//
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

#ifdef INSTRUCTION_BASE
#else
include "instruction_base.td"
#endif

let cat_ = cat_controlFlow in {
  def Loop: Inst <"Generate a loop."> {
    let attrs_ = [ Attr<"The loop body run each iteration.", BasicBlockPtr,
                        "body", "nullptr"> ];
    let ins_ = [
      Arg<"The maximum loop count.", ArgType<[ I64 ]>, 0D>,
      Arg<"The termination condition.", ArgType<[ I1 ]>, 0D>,
      VarArg<"The initial values across each loop iterations.",
             ArgType<[ I8, I16, I32, I64, F16, F32 ]>, 0D>
    ];
    let outs_ =
        [ Arg<"The result.", ArgType<[ I8, I16, I32, I64, F16, F32 ]> > ];
  }

  def If: Inst <"Generate if conditional."> {
    let attrs_ = [
      Attr<"Else branch.", BasicBlockPtr, "else_branch", "nullptr">,
      Attr<"Then branch.", BasicBlockPtr, "then_branch", "nullptr">
    ];
    let ins_ = [ Arg<"The condition for if.", ArgType<[ I1 ]>, 0D> ];
    let outs_ =
        [ Arg<"The result.", ArgType<[ I8, I16, I32, I64, F16, F32 ]> > ];
  }

  def Jump: Inst <"Generate unconditional jump."> {
    let attrs_ = [
      Attr<"Jump target address.", BasicBlockPtr, "target", "nullptr">
    ];
    let ins_ = [
      Arg<"The condition for jump, this value is always true in unconditional case.",
           ArgType<[ I1 ]>, 0D> ];
    let outs_ =
        [ Arg<"The result.", ArgType<[ I8, I16, I32, I64, F16, F32 ]> > ];
  }
}