//===- loss_instructions.td ----------------------------------*- tblgen -*-===//
//
// Copyright (C) 2019-2021 Alibaba Group Holding Limited.
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

let cat_ = cat_loss in {
  let attrs_ = [ Attr<"ignored target value.", Integer, "ignored">,
                 Attr<"reduction algorithm.", EnumReduction, "reduction"> ],
      ins_ = [Arg<"The input data", ArgType<[F16,F32, I32, I64]>>,
      	Arg<"Ground truth", ArgType<[I32, I64]>>,
        OptionalArg<"Weights", MatchArgType<0>>
      ]
   in {
    def NegativeLogLikelihoodLoss : Inst<"Negative Log Likelihood Loss."> {
      let outs_ = [Arg<"The result.", MatchArgType<0> >];
    }

    def SoftmaxCrossEntropyLoss: Inst<"Softmax cross entropy Loss."> {
      let outs_ = [
	      Arg<"The result.", MatchArgType<0> >,
        OptionalArg<"Log proability", MatchArgType<0>>
      ];
    }

  }
}