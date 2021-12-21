//===- common_reduction_instructions.td ----------------------*- tblgen -*-===//
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

let cat_ = cat_common_reduction,
    attrs_ = [Attr<"Axis along which to reduce, omitted if X2 exists.",
                   IntegerList, "axis">,
              Attr<"Indicate whether to keep the reduced dimension.",
                   Bool, "keep_dims", "true">,
              Attr<"Use to avoid division by zero.",
                   Float, "epsilon", "0">,
    ],
    ins_ = [Arg<"The input.", ArgType<[I8,I16,I32,F16,F32]> >,
              OptionalArg<"The axis.", ArgType<[I32]>, 1D>],
    outs_ = [Arg<"The result", MatchArgType<0> >] in {
      def ReduceL1 : Inst<"Computes the L1 norm of the elements across"
                        " dimensions.">;
      def ReduceL2 : Inst<"Compute the L2 norm of the elements across"
                        " dimensions.">;
    }

let cat_ = cat_common_reduction,
    attrs_ = [Attr<"Axis along which to reduce, omitted if X2 exists.",
                   IntegerList, "axis">,
              Attr<"Indicate whether to keep the reduced dimension.",
                   Bool, "keep_dims", "true">],
    ins_ = [Arg<"The input.", ArgType<[I8,I16,I32,F16,F32]> >,
              OptionalArg<"The axis.", ArgType<[I32]>, 1D>],
    outs_ = [Arg<"The result", MatchArgType<0> >] in {
    def ReduceLogSum : Inst<"Compute the log sum of the elements across"
                            " dimensions.">;

    def ReduceLogSumExp : Inst<"Compute the log sum exp of the elements across"
                               " dimensions.">;

    def ReduceSumSquare : Inst<"Compute the log sum of the elements across"
                               " dimensions.">;

    def ReduceMax : Inst<"Compute the maximum of the elements across"
                         " dimensions">;

    def ReduceMean : Inst<"Compute the average of the elements across"
                          " dimensions.">;

    def ReduceMin : Inst<"Compute the minimum of the elements across"
                         " dimensions">;

    def ReduceProduct : Inst<"Compute the product of the elements across"
                             " dimensions">;

    def ReduceSum : Inst<"Compute the sum of the elements across"
                         " dimensions">;

  /* def ReduceAll : Inst<"Compute the logical AND of the elements across"
                       " dimensions."> {
    let ins_ = [Arg<"The input.", ArgType<[I1]> >,
                OptionalArg<"The axis.", ArgType<[I32]>, 1D>];
  }*/
}

let cat_ = cat_common_reduction,
    attrs_ = [Attr<"Axis along which to reduce.", Integer, "axis", "0">,
              Attr<"Indicate whether to keep the reduced dimension.",
                   Bool, "keep_dims", "true">,
              Attr<"Select the last occurrence of the value if multiple candidates are available",
                    Bool, "select_last_index", "false">],
    ins_ = [Arg<"The input.", ArgType<[I8,I16,I32,F16,F32]> >,
            OptionalArg<"The axis.", ArgType<[I32]>, 0D>],
    outs_ = [Arg<"The result", ArgType<[I64]> >] in {

  def Argmax : Inst<"Computes the indices of the largest value of the input"
                    " along the specified axis.">;

  def Argmin : Inst<"Computes the indices of the smallest value of the input"
                    " along the specified axis.">;
}

let cat_ = cat_common_reduction in {
  def CumSum : Inst<"Computes cumulative sum of the input alone the specified axis."> {
    let attrs_ = [
      Attr<"Axis along which to do accumulation.", Integer, "axis", "0">,
      Attr<"Exclude the first element from sum.", Bool, "exclusive", "false">,
      Attr<"Reverse the direction of accumulation.", Bool, "reverse", "false">
    ];
    let ins_ = [Arg<"The input.", ArgType<[I8,I16,I32,F16,F32,F64]> >,
            OptionalArg<"The axis.", ArgType<[I32]>, 0D>];
    let outs_ = [Arg<"The result", MatchArgType<0> >];
  }
}
