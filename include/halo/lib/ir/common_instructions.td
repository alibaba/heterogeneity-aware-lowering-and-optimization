//===- common_instructions.td --------------------------------*- tblgen -*-===//
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

let cat_ = cat_common in {

  def Concat : Inst <"Concatenate a list of inputs along axis."> {
    let attrs_ = [Attr<"The axis to concatenate on.", Integer, "axis", "0">,
                  Attr<"The number of inputs.", Integer, "N", "0">];
    let ins_ = [VarArg<"The list of inputs.", ArgType<[I8,I16,I32,F16,F32]> >,
                OptionalArg<"The axis.", ArgType<[I32]>, 0D>];
    let outs_ = [Arg<"The result.", MatchArgType<0> >];
  }

  def Compress : Inst<"Select slices of input alone axis."> {
    let attrs_ = [Attr<"The axis to select from", Integer, "axis", "std::numeric_limits<int>::max()">];
    let ins_ = [Arg<"The inpput data.", ArgType<[I8,I16,I32,F16,F32]> >,
                Arg<"The conditions.", ArgType<[I1]> >];
    let outs_ = [Arg<"The result.", MatchArgType<0> >];
  }

  def OneHot : Inst <"Generate a one-hot NDim array."> {
    let attrs_ = [Attr<"The axis to fill data in the result.",
                       Integer, "axis", "-1">];
    let ins_ = [Arg<"Indices to fill in the on-value X2", ArgType<[I32, I64]> >,
                Arg<"The size of the one hot dimension specified by axis",
                    ArgType<[I32]>, 0D>,
                Arg<"The on-value.", ArgType<[I8,I16,I32,F16,F32]>, 0D>,
                Arg<"The off-value.", MatchArgType<2>, 0D>];
    let outs_ = [Arg<"The result.", MatchArgType<2> >];
  }

  def Gather : Inst <"Gather data from input X1 along axis based on X2."> {
    let attrs_ = [Attr<"The axis to gather on", Integer, "axis", "0">];
    let ins_ = [Arg<"The inpput data.", ArgType<[I8,I16,I32,F16,F32]> >,
                Arg<"The indices.", ArgType<[I32, I64]> >,
                OptionalArg<"The axis.", ArgType<[I32]>, 0D>];
    let outs_ = [Arg<"The result.", MatchArgType<0> >];
  }

  def GatherElements : Inst <"Gather data from input X1 based on indices specified by X2."> {
    let attrs_ = [Attr<"The axis to gather on", Integer, "axis", "0">];
    let ins_ = [Arg<"The input data.", ArgType<[I8,I16,I32,F16,F32]> >,
                Arg<"The indices.", ArgType<[I32, I64]> >];
    let outs_ = [Arg<"The result.", MatchArgType<0> >];
  }

  def Pad : Inst<"Pad the input data based on various algorithms specified by"
                 " modes along each spatial axis."> {
    let attrs_ = [Attr<"The padding algorithm such as CONSTANT, SYMMETRIC,"
                       " REFLECT.",
                  EnumPadMode, "mode", "CONSTANT">];
    let ins_ = [Arg<"The input data.", ArgType<[I8,I16,I32,F16,F32]> >,
                Arg<"The padding data indicating how many elements to add"
                    "  before and after each axis of input, it has a shape of"
                    "  [R, 2], where R is the rank of input.",
                    ArgType<[I32]>, 2D>,
                OptionalArg<"The padding value used when modes is CONSTANT",
                            MatchArgType<0>, 0D>];
    let outs_ = [Arg<"The result.", MatchArgType<0> >];
  }

  def Shape : Inst<"Compute the shape of input."> {
    let ins_ = [Arg<"The input.", ArgType<[I8,I16,I32,F16,F32]> >];
    let attrs_ = [Attr<"The output date type size", EnumDataType, "data_type", "INT64">];
    let outs_ = [Arg<"The result.", ArgType<[I64, I32]> >];
  }

  def Reshape : Inst<"Reshape the input X1 to create the result with the same"
                     " number of elements and the shape specified by X2."> {
    let ins_ = [Arg<"The input.", ArgType<[I8,I16,I32,F16,F32]> >,
                Arg<"The shape.", ArgType<[I32,I64]>, 1D>];
    let outs_ = [Arg<"The result.", MatchArgType<0> >];
  }

  def Return : Inst<"Specify the final output results."> {
    let ins_ = [VarArg<"The operands that will be used as final results",
                    ArgType<[I8,I16,I32,F16,F32]> >];
  }

  def Call : Inst<"Call a Halo Function"> {
    let ins_ = [VarArg<"The arguments passed to callee",
                ArgType<[I8,I16,I32,F16,F32]> >];
    let attrs_ = [Attr<"The callee function", FunctionPtr, "callee", "nullptr">];
  }

  def SetDiff1D : Inst<"Return the data from X1 if it is not in X2 and preserve"
                       " the original order."> {
    let ins_ = [Arg<"Values to keep.",
                    ArgType<[I8,I16,I32,F16,F32]>, 1D>,
                Arg<"Values to remove.", MatchArgType<0>, 1D>];
    let outs_ = [Arg<"The result.", MatchArgType<0>, 1D>];
  }

  def Slice : Inst<"Extract a slice from the input."> {
    let ins_ = [Arg<"The input.", ArgType<[I8,I16,I32,F16,F32]> >,
                Arg<"The starting indices, inclusive, along the specified"
                    " axes.",
                    ArgType<[I32]>, 1D>,
                Arg<"The size of the slice, along the specified axes.",
                    ArgType<[I32]>, 1D>,
                OptionalArg<"The slice steps along the specified axes.",
                    ArgType<[I32]>, 1D>,
                OptionalArg<"The axes", ArgType<[I32]>, 1D>];
    let outs_ = [Arg<"The result.", MatchArgType<0> >];
  }

  def SliceDynamic : Inst<"Extract a dynamic slice from the input."> {
    let ins_ = [Arg<"The input.", ArgType<[I8,I16,I32,F16,F32]> >,
                Arg<"The starting indices, inclusive, along the specified"
                    " axes.",
                    ArgType<[I32]>, 1D>,
                Arg<"The size of the slice, along the specified axes.",
                    ArgType<[I32]>, 1D>,
                OptionalArg<"The slice steps along the specified axes.",
                    ArgType<[I32]>, 1D>,
                OptionalArg<"The axes", ArgType<[I32]>, 1D>];
    let outs_ = [Arg<"The result.", MatchArgType<0> >];
  }

  def Split : Inst<"Splits a tensor into num_split tensors along one dimension."> {
    let attrs_ = [Attr<"The number of ways to split. ", Integer, "num_split", "1">];
    let ins_ = [Arg<"The dimension along which to split.", ArgType<[I32,I64]> >,
                Arg<"The input.", ArgType<[I8,I16,I32,F16,F32]> >];
  }

  def Stack : Inst<"Join a list of NDArray inputs along a new axis"> {
    let attrs_ = [Attr<"The axis to stack on", Integer, "axis", "0">];
    let ins_ = [VarArg<"The list of inputs.", ArgType<[I8,I16,I32,F16,F32]> >];
    let outs_ = [Arg<"The result.", MatchArgType<0> >];
  }

  def Tile : Inst<"Replicate input multiples times."> {
    let ins_ = [Arg<"The input.", ArgType<[I8,I16,I32,F16,F32]> >,
                Arg<"The multiples.", ArgType<[I32,I64]>, 1D>];
    let outs_ = [Arg<"The result.", MatchArgType<0> >];
  }

  def TileDynamic : Inst<"Replicate input multiples times dynamically."> {
    let ins_ = [Arg<"The input.", ArgType<[I8,I16,I32,F16,F32]> >,
                Arg<"The multiples.", ArgType<[I32,I64]>, 1D>];
    let outs_ = [Arg<"The result.", MatchArgType<0> >];
  }

  def ExpandDims: Inst<"Broadcast the input following the given shape"> {
    let ins_ = [Arg<"The input.", ArgType<[I8,I16,I32,F16,F32]> >,
                Arg<"The shape.", ArgType<[I32,I64]>, 1D>];
    let outs_ = [Arg<"The result.", MatchArgType<0> >];
  }

  def NoOp: Inst<"No Op"> {
    let ins_ = [Arg<"The input.", ArgType<[I8,I16,I32,F16,F32]> >];
    let outs_ = [Arg<"The result.", MatchArgType<0> >];
  }

  def Select: Inst<"Tenary selection"> {
    let ins_ = [
      Arg<"The condition.", ArgType<[I1]>>,
      Arg<"The true value.", ArgType<[I1,I8,I16,I32,I64,F16,F32,F64]>>,
      Arg<"The false value.", MatchArgType<1>>,
    ];
    let outs_ = [Arg<"The result.", MatchArgType<1>>];
  }

  def Unique: Inst<"Find the unique elements of the input."> {
    let attrs_ = [Attr<"Result indices' type.",
                  EnumDataType, "out_idx_type", "INVALID">];
    let ins_ = [Arg<"The input.", ArgType<[I8,I16,I32,F16,F32]> >];
    let outs_ = [Arg<"The unique elements of the input.",
                     MatchArgType<0> >,
                 Arg<"The indices of the unique elements",
                 ArgType<[I32,I64]>, 1D>];
  }

}