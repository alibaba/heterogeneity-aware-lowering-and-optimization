//===- hgengine.cc
//----------------------------------------------------------===//
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

#include <string>

#include "halo/lib/ir/common_instructions.h"
#include "halo/lib/target/generic_cxx/generic_cxx_codegen.h"

namespace halo {

void GenericCXXCodeGen::RunOnInstruction(HgQuantInst* inst) {

  const Def& input = inst->GetOperand(0);

  CXXValue op0 = ir_mapping_[input];
  
  const std::string& enum_ns_layout = "odla_memory_layout::";
  const std::string& enum_prefix = "ODLA_";
  const std::string& enum_ns =
      opts_.dialect == Dialect::CXX_11 ? enum_ns_layout : "";

  /// By now the Hgai onnx interfce set in_scale/in_bias to string
  std::vector<double> in_scale;
  std::vector<double> in_bias;
  in_scale.reserve(1);
  in_bias.reserve(1);

  in_scale.emplace_back(std::stof(inst->GetInScale()));
  in_bias.emplace_back(std::stof(inst->GetInBias()));
  std::string qtype = inst->GetQtype();
  int is_per_channel = inst->GetIsPerChannel();

  const std::string& input_layout =
      enum_ns + enum_prefix +
      (inst->GetInDataFormat() == "NHWC" ? "CHANNELS_LAST" : "CHANNELS_FIRST");
  const std::string& output_layout =
      enum_ns + enum_prefix +
      (inst->GetOutDataFormat() == "NHWC" ? "CHANNELS_LAST" : "CHANNELS_FIRST");

  // Todo : transpose support, per_channel support, int16/fp16 support
  HLCHECK(input_layout == output_layout);  
  HLCHECK(is_per_channel == false);
  HLCHECK(qtype == "int8");

  std::vector<CXXValue> inputs;
  const Def& op = inst->GetOperand(0);
  CXXValue op_v = ir_mapping_[op];
  inputs.push_back(op_v);

  std::vector<CXXValue> rets;
  rets.emplace_back(inst->GetName() + std::to_string(0),
                      TensorTypeToCXXType(inst->GetResultType(0), false));
  ir_mapping_[Def(inst, 0)] = rets[0];


  // construct the odla_value_ids string
  unsigned int id = 0;
  std::ostringstream os;
  os << "{.size = " << inputs.size() << ", .value_ids = {";
  for (auto& one : inputs) {
    os << "(const odla_value_id)";
    EmitODLAVauleId(one, os);
    if (++id != inputs.size()) {
      os << ", ";
    }
  }
  os << "}}";

  EmitODLACall<2, false>(rets, "odla_CustomOp", inputs, "\"HgaiQuant\"",
                         "\"" + inst->GetName() + "\"", os.str(), input_layout,
                         output_layout, is_per_channel,
                          "\"" + qtype + "\"", in_scale[0], in_bias[0]);
}

} // namespace halo
