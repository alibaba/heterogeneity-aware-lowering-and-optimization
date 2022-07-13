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

  def HgEngine : Inst<"Hg engine op."> {
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

  def HgQuant : Inst<"Hg Quant op."> {
    let attrs_ = [Attr<"qtype", String, "qtype", "\"int8\"">,
                  Attr<"is_per_channel", Integer, "is_per_channel", "0">,
                  Attr<"int data format, default is NHWC", String, "in_data_format", "\"NCHW\"" >,
                  Attr<"out data format, default is NHWC", String, "out_data_format", "\"NCHW\"" >,
                  Attr<"in_type", String, "in_type", "\"float32\"" >,
                  Attr<"out_type", String, "out_type", "\"int8\"" >,
                 ];
    let ins_ = [Arg<"Input data.", ArgType<[F16,F32]> >];
    let outs_ = [Arg<"The result Quantized value.", ArgType<[I8,I16]> >];
  }

  def HgDequant : Inst<"Hg Dequant op."> {
    let attrs_ = [Attr<"is_per_channel", Integer, "is_per_channel", "0">,
                  Attr<"int data format, default is NHWC", String, "in_data_format", "\"NCHW\"" >,
                  Attr<"out data format, default is NHWC", String, "out_data_format", "\"NCHW\"" >,
                  Attr<"in_type", String, "in_type", "\"int8\"" >,
                  Attr<"out_type", String, "out_type", "\"float32\"" >,
                 ];
    let ins_ = [Arg<"Input data.", ArgType<[I8,I16]> >];
    let outs_ = [Arg<"The result DeQuantized value.", ArgType<[F32,F16]> >];
  }

  def KvParser : Inst<"KVParser op"> {
    let attrs_ = [Attr<"common flag", String, "common_flag", "\"[common]\"">,                     
                  Attr<"dense column names", StringList, "dense_column_names", "{\"\"}">,
                  Attr<"dense default values", StringList, "dense_default_values", "{\"\"}">,
                  Attr<"dense max lens", StringList, "dense_max_lens", "{\"\"}">,
                  Attr<"dense need actual lens", BoolList, "dense_need_actual_lens", "{false}">,
                  Attr<"dense separators", StringList, "dense_separators", "{\",\"}">,
                  Attr<"dense types", StringList, "dense_types", "{\"\"}">,                       /// ? To check if TypeList is available.
                  Attr<"empty value valid", Bool, "empty_value_valid", "false">,
                  Attr<"Nsparse", Integer, "Nsparse", "0">,
                  Attr<"Nvar", Integer, "Nvar", "0">,
                  Attr<"record flag", String, "record_flag", "\"[dat]\"">, 
                  Attr<"sparse column names", StringList, "sparse_column_names", "{\"\"}">,       /// Not used in BERT like model
                  Attr<"sparse index columns", IntegerList, "sparse_index_columns", "">,          /// Not used in BERT like model
                  Attr<"sparse index value separators", StringList, "sparse_index_value_separators", "{\"\"}">, /// Not used in BERT like model
                  Attr<"sparse_max_lens", IntegerList, "sparse_max_lens", "">,                    /// Not used in BERT like model
                  Attr<"sparse_separators", StringList, "sparse_separators", "{\"\"}">,           /// Not used in BERT like model
                  Attr<"sparse_types", StringList, "sparse_types", "{\"\"}">,                     /// Not used in BERT like model
                  Attr<"sparse_value_columns", StringList, "sparse_value_columns", "{\"\"}">,     /// Not used in BERT like model
                  Attr<"var_len_column_names", StringList, "var_len_column_names", "{\"\"}">,
                  Attr<"var_len_max_lens", StringList, "var_len_max_lens", "{\"\"}">,
                  Attr<"var_len_separators", StringList, "var_len_separators", "{\",\"}">,
                  Attr<"var_len_types", StringList, "var_len_types", "{\"\"}">                   /// ? To check if TypeList is available.
                  ];
    let ins_ = [Arg<"Input data.", ArgType<[Str]> >];
    ///? Not determine outs_ because the number and type of output are not determined.
  }
}
