//===- customize_instrucitons.td ----------------------*- tblgen -*-===//
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

let cat_ = cat_customize in {

  def HgEngine : Inst<"Alinpu engine op."> {
    let attrs_ = [Attr<"serialized engine string", String, "serialized_engine", "\"NHWC\"">,
                  Attr<"model name", String, "model_name", "\"\"">,
                  Attr<"op name", String, "op_name", "\"\"">,
                  Attr<"global scheme", String, "global_scheme", "\"\"">,
                  Attr<"int data format, default is NHWC", String, "in_data_format", "\"NHWC\"" >,
                  Attr<"out data format, default is NHWC", String, "out_data_format", "\"NHWC\"" >,
                  Attr<"in binding list", StringList, "in_binding_list", "{\"\"}">,
                  Attr<"out binding list", StringList, "out_binding_list", "{\"\"}">,
                  Attr<"output shapes info ", IntegerListList, "output_shapes", "">,
                  Attr<"in type list, like int8", StringList, "in_type_list", "{\"int8\"}">,
                  Attr<"out type list, like int8", StringList, "out_type_list", "{\"int8\"}">,
                  Attr<"device id", Integer, "device_id", "-1">
                 ];
    let ins_ = [Arg<"Input data.", ArgType<[I8, I16]> >];
    let outs_ = [Arg<"The result value.", MatchArgType<0>>];
  }

}
