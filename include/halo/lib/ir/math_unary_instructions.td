//===- math_unary_instructions.td ----------------------------*- tblgen -*-===//
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

let cat_ = cat_math_unary,
    ins_ = [Arg<"The unary input.", ArgType<[I8,I16,I32,F16,F32]> >],
    outs_ = [Arg<"The result.", MatchArgType<0> >] in {

  def Abs : Inst<"The unary opertion returns abs(X1), element-wise.">;

  def Erf : Inst<"The unary operation returns the error function of X1,"
                 " element-wise.">;

  def Neg : Inst<"The unary opertion returns -X1, element-wise.">;

  def Sign: Inst<"The unary opertion returns 1 if X1 > 0, return -1 if X1 < 0,"
                 " return 0 if X1 == 0, element-wise.">;

}

let cat_ = cat_math_unary,
    ins_ =  [Arg<"The unary input.", ArgType<[F16,F32]> >],
    outs_ = [Arg<"The result.", MatchArgType<0> >] in {

  def Ceil : Inst<"The unary opertion returns ceil(X1), element-wise.">;

  def Exp  : Inst<"The unary opertion returns exp(X1), element-wise.">;

  def Floor: Inst<"The unary opertion returns floor(X1), element-wise.">;

  def Rcp  : Inst<"The unary opertion returns 1 / X1, element-wise.">;

  def Rsqrt : Inst<"The unary opertion returns 1 / (X1 ^ 0.5), element-wise.">;

  def Sqrt : Inst<"The unary opertion returns X1 ^ 0.5, element-wise.">;

}