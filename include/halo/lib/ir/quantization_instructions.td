//===- quantization_instructions.td --------------------------*- tblgen -*-===//
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

let cat_ = cat_quantization in {
  def Quantize: Inst<"Quantize a tensor."> {
    let ins_ = [Arg<"The input.", ArgType<[F32]>>,
                Arg<"Scale for input.", ArgType<[F32]>>,
                Arg<"Zero point for input.", ArgType<[F32,I8,I16]>>
               ];
    let attrs_ = [ Attr<"axis.", Integer, "axis", "0">,
                   Attr<"bits used for quantized tensor.", Integer, "bits", "8">,
                   Attr<"use sign bit for quantized tensor.", Bool, "sign_bit", "false"> ];
    let outs_ = [Arg<"The result.", ArgType<[I8, I16, I32]>>];
  }

  def DynamicQuantize: Inst<"Dynamically quantize a tensor."> {
    let ins_ = [Arg<"The input.", ArgType<[F32]>>];
    let attrs_ = [ Attr<"axis.", Integer, "axis", "0">,
                   Attr<"bits used for quantized tensor.", Integer, "bits", "8">,
                   Attr<"use sign bit for quantized tensor.", Bool, "sign_bit", "false"> ];
    let outs_ = [Arg<"The result.", ArgType<[I8, I16, I32]>>,
                 Arg<"scale when computed online", ArgType<[F32]>>,
                 Arg<"zero point when computed online", ArgType<[I8, I16]>>,
                ];
  }

  def Dequantize: Inst<"Dequantize a tensor."> {
    let ins_ = [Arg<"The input.", ArgType<[I8, I16, I32]>>,
                Arg<"Scale for input.", ArgType<[F32]>>,
                OptionalArg<"Zero point for input.", ArgType<[F32]>>
               ];
    let attrs_ = [ Attr<"axis.", Integer, "axis", "0"> ];
    let outs_ = [Arg<"The result.",ArgType<[F32]> >];
  }
}