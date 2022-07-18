//===- math_binary_instructions.td ---------------------------*- tblgen -*-===//
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

let cat_ = cat_math_binary,
    ins_ = [Arg<"The left hand side operand.", ArgType<[I8,I16,I32,F16,F32]> >,
            Arg<"The right hand side operand", MatchArgType<0> >],
    outs_ = [Arg<"The result.", MatchArgType<0> >] in {

  def Add : Inst<"Element-wise addition, return X1 + X2, support broadcast.">;

  def Div : Inst<"Element-wise division, return X1 / X2, support broadcast.">;

  def Maximum : Inst<"Compute maximum value of the inputs, element-wise,"
                     " return max(X1, X2), support broadcast.">;

  def Mean : Inst<"Compute mean value of the inputs, element-wise,"
                     " return mean(X1, X2), support broadcast.">;

  def Minimum : Inst<"Compute minimum value of the inputs, element-wise,"
                     " return min(X1, X2), support broadcast.">;

  def Mul : Inst<"Binary multiplication, element-wise, return X1 * X2,"
                 " support broadcast.">;

  def Pow : Inst<"Power operation, element-wise, return X1 ^ X2, support"
                 " broadcast.">;

  def SquaredDifference : Inst<"Returns conj(x - y)(x - y) element-wise.">;

  def Sub : Inst<"Binary subtraction operation, element-wise,"
                 " return X1 - X2, support broadcast.">;

}

def Mod : Inst<"Modulo operation, element-wise, support broadcast."> {
   let cat_ = cat_math_binary;
   let attrs_ = [ Attr<"fmod", Integer, "fmod", "0"> ];
   let ins_ = [
     Arg<"The left hand side operand.", ArgType<[ I8, I16, I32, F16, F32, F64 ]> >,
     Arg<"The right hand side operand", MatchArgType<0> >
   ];
   let outs_ = [ Arg<"The result.", MatchArgType<0> >];
 }

let cat_ = cat_math_binary,
    ins_ = [Arg<"The left hand side operand.", ArgType<[I8,I16,I32]> >,
            Arg<"The right hand side operand", MatchArgType<0> >],
    outs_ = [Arg<"The result.", MatchArgType<0> >] in {

  def ShiftL : Inst<"Bitwise shift left operation, element-wise,"
                    " return X1 << X2, support broadcast.">;

  def ShiftR : Inst<"Bitwise logical shift right operation, element-wise,"
                    " appending zeros as MSB, return X1 >> X2, support"
                    " broadcast.">;

}

let cat_ = cat_math_binary,
    ins_ = [Arg<"The left hand side operand.", ArgType<[I1]> >,
            Arg<"The right hand side operand", ArgType<[I1]> >],
    outs_ = [Arg<"The result.", ArgType<[I1]> >] in {

  def And : Inst<"Logical AND operation, element-wise, return X1 && X2,"
                 " support broadcast.">;

  def Or : Inst<"Logical OR operation, element-wise, return X1 || X2,"
                " support broadcast.">;

  def Xor : Inst<"Logical XOR operation, element-wise, return X1 != X2,"
                " support broadcast.">;
}

def Cmp : Inst<"Compare operation, element-wise, support broadcast."> {
  let cat_ = cat_math_binary;
  let attrs_ = [ Attr<"predicator", EnumPred, "predicator", "EQ"> ];
  let ins_ = [
    Arg<"The left hand side operand.", ArgType<[ I8, I16, I32, F16, F32 ]> >,
    Arg<"The right hand side operand", MatchArgType<0> >
  ];
  let outs_ = [ Arg<"The result.", ArgType<[ I1 ]> > ];
}

def Shift : Inst<"Element-wise bit shift."> {
  let cat_ = cat_math_binary;
  let attrs_ = [ Attr<"is left shift", Bool, "is_left_shift", "true">];
  let ins_ = [
    Arg<"The left hand side operand.", ArgType<[ I8, I16, I32 ]> >,
    Arg<"The right hand side operand", MatchArgType<0> >
  ];
}