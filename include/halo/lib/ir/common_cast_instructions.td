//===- common_cast_instructions.td ---------------------------*- tblgen -*-===//
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

let cat_ = cat_common_cast,
    attrs_ = [Attr<"The datatype to which the input data are cast",
                   EnumDataType, "data_type", "INVALID">],
    ins_ = [Arg<"The input.", ArgType<[I8,I16,I32,F16,F32]> >],
    outs_ = [Arg<"The result", ArgType<[I8,I16,I32,F16,F32]> >] in {

  def SItoFP : Inst<"Cast the element of input X1 from signed integer"
                    "to floating point type">;
  def FPtoSI : Inst<"Cast the element of input X1 from floating point"
                    "to the integer type">;
  def ZExt : Inst<"Perform zero-extension on X1">;

  def FPtoFP : Inst<"Cast the element of input X1 between different floating point format">;

  def ConvertFromString: Inst<"Parse input strings into specified data type">;

  def ConvertToString: Inst<"Convert input data into string">;

  def Bitcast: Inst<"Bitcast input tensor to another type.">;
}