//===- creator_instructions.td -------------------------------*- tblgen -*-===//
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

let cat_ = cat_creator in {

  def Range : Inst <"Generate a sequence of numbers, starting from X1, up to"
                    " X2, by increments of X3."> {
    let ins_ = [Arg<"The first number of the result.",
                    ArgType<[I8,I16,I32,F16,F32]>, 0D>,
                Arg<"The exclusive limit of the result.",
                    MatchArgType<0>, 0D>,
                Arg<"The incrementing value.", MatchArgType<0>, 0D>];
    let outs_ = [Arg<"The result.", MatchArgType<0>, 1D>];
  }

  def RandomUniform : Inst<"Generate an NDim array with random numbers to form a"
                           " uniform distribution."> {
    let attrs_ = [Attr<"The result data type", EnumDataType, "dtype", "FLOAT32">,
                  Attr<"The lower bound of the result value, inclusive.",
                       Float, "minval", "0.0">,
                  Attr<"The upper bound of the result value, exclusive.",
                       Float, "maxval", "1.0">,
                  Attr<"The result shape.", IntegerList, "shape", "{}">,
                  Attr<"The seed to the random generator.", Integer, "seed", "0">];
    let ins_ = [OptionalArg<"The result shape.", ArgType<[I32]>, 1D>];
    let outs_ = [Arg<"The result.", ArgType<[I8,I16,I32,F16,F32]> >];
  }

}