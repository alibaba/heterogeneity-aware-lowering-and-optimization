//===- math_instructions.td ----------------------------------*- tblgen -*-===//
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

let cat_ = cat_math in {

  def BatchMatMul : Inst<"Batched matrix product such that each slice of X1 and"
                         " X2 perform MatMul and the result has the same batch"
                         " size as input."> {
    let attrs_ = [Attr<"whether X1 needs transpose.",
                       Bool, "transpose_a", "false">,
                  Attr<"whether X2 needs transpose.",
                       Bool, "transpose_b", "false">];
    let ins_ = [Arg<"Shape of (b1, ..., bm, M, K), or (b1, ..., bm, K, M) if"
                    " transpose_a is true.",
                    ArgType<[I8,I16,I32,F16,F32]> >,
                Arg<"Shape of (b1, ..., bm, K, N), or (b1, ..., bm, N, K) if"
                    " transpose_b is true.",
                    MatchArgType<0> >];
    let outs_ = [Arg<"The result of the matrix multiplication.",
                     MatchArgType<0> >];
  }

  def Gemm : Inst<"General matrix multiplication, result value is alpha * X1"
                  " * X2 + beta * X3, where X1= transpose(X1) if transA"
                  " else X1, X2 = transpose(X2) if transB else X2."> {
    let attrs_ = [Attr<"multiplier of X1 * X2", Float, "alpha", "1.0">,
                  Attr<"multiplier of X3", Float, "beta", "1.0">,
                  Attr<"whether X1 needs transpose",
                       Bool, "transpose_a", "false">,
                  Attr<"whether X2 needs transpose",
                       Bool, "transpose_b", "false">];
    let ins_ = [Arg<"2D Array of shape (M, K), "
                    "or (K, M) if transA is non-zero.",
                    ArgType<[I8,I16,I32,F16,F32]>, 2D>,
                Arg<"2D Array of shape (K, N), "
                    "or (N, K) if transB is non-zero.",
                    MatchArgType<0>, 2D>,
                Arg<"2D Array of shape (M, N).", MatchArgType<0>, 2D>];
    let outs_ = [Arg<"The result of shape (M, N).", MatchArgType<0>, 2D>];
  }

  def InnerProduct : Inst <"Calculate inner product of two inputs, optionally"
                           " add bias. The result value is  of X1 * X2' + X3,"
                           " where X2' = transpose(X2) if transpose else X2."
                          > {
    let attrs_ = [Attr<"whether X2 needs transpose.",
                       Bool, "transpose", "false">,
                  Attr<"the size of the result.", Integer, "num_output", "0">,
                  Attr<"the first axis to be lumped into a single inner"
                       " product computation.",
                       Integer, "axis", "0">,
                  Attr<"whether input X1 needs to be flattened.",
                       Integer, "flatten", "0">];
    let ins_ = [Arg<"Filter operand.", ArgType<[I8,I16,I32,F16,F32]> >,
                Arg<"Weight operand.", MatchArgType<0> >,
                OptionalArg<"Bias operand.", MatchArgType<0> >];
    let outs_ = [Arg<"The result value.", MatchArgType<0> >];
  }

  def MatMul : Inst <"2D Matrix product, the result is computed as X1' * X2'"
                     " where X1' = transpose(X1) if trans_a, else X1' = X1;"
                     " and X2' = transpose(X2) if trans_b else X2' = X2."> {
    let attrs_ = [Attr<"whether X1 needs transpose.",
                       Bool, "transpose_a", "false">,
                  Attr<"whether X2 needs transpose.",
                       Bool, "transpose_b", "false">];
    let ins_ = [Arg<"Shape of (M, K), or (K, M) if transpose_a is true.",
                    ArgType<[I8,I16,I32,F16,F32]>, 2D>,
                Arg<"Shape of (K, N), or (N, K) if transpose_b is true.",
                    MatchArgType<0>, 2D>];
    let outs_ = [Arg<"The result of the matrix multiplication.",
                     MatchArgType<0>, 2D>];
  }

  def Transpose : Inst<"Transpose the input such that the dimension is"
                       "  permutated based on permutation."> {
    let attrs_ = [Attr<"permutation of the dimension indices.",
                       IntegerList, "permutation">];
    let ins_ = [Arg<"The input.", ArgType<[I8,I16,I32,F16,F32]> >,
                OptionalArg<"The permutation.", ArgType<[I32]>, 1D>];
    let outs_ = [Arg<"The result.", MatchArgType<0> >];
  }

  def Det : Inst<"Det calculates determinant of a square matrix or "
                       "batches of square matrices."> {
     let ins_ = [Arg<"Input tensor", ArgType<[F16,F32,F64]> >];
     let outs_ = [Arg<"Input tensor", MatchArgType<0> >];
  }

  def Einsum: Inst<"Einsum calculates the Einstein summation"> {
     let ins_ = [Arg<"Input tensor", ArgType<[F16,F32,F64]> >];
     let outs_ = [Arg<"Input tensor", MatchArgType<0> >];
  }

}