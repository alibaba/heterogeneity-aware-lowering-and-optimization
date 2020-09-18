//===- call.cc ------------------------------------------------------------===//
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

#include "halo/lib/ir/ir_builder.h"
#include "halo/lib/target/generic_cxx/generic_cxx_codegen.h"

namespace halo {

static const std::string& GetDeviceVariable(const Function& func) {
  static const std::string default_name = "ODLA_DEVICE_DEFAULT";
  static const std::unordered_map<std::string, std::string> names{
      {"x86", "x86_dev"}, {"tensorrt", "trt_dev"}};
  auto it = names.find(func.GetDeviceName());
  if (it == names.end()) {
    return default_name;
  }
  return it->second;
}

void GenericCXXCodeGen::RunOnInstruction(CallInst* inst) {
  int nr_rets = inst->GetNumOfResults();

  std::vector<CXXValue> outputs(nr_rets);
  for (int i = 0; i < nr_rets; ++i) {
    const auto& type = inst->GetResultType();
    CXXValue ret(inst->GetName(), TensorTypeToCXXType(type, false));
    EmitODLACall(ret, "odla_CreateValue", type);
    outputs[i] = ret;
    ir_mapping_[Def(inst, i)] = ret;
  }

  std::vector<CXXValue> inputs;
  inputs.reserve(inst->GetOperands().size());
  for (auto& op : inst->GetOperands()) {
    inputs.push_back(ir_mapping_[op]);
  }

  os_ << "  odla_RunTaskAsync(" << GetDeviceVariable(*inst->GetCallee())
      << ", (odla_task){" << inst->GetCallee()->GetName() << ", .inputs=";
  EmitODLAArgs(inputs);
  os_ << ", .outputs=";
  EmitODLAArgs(outputs);
  os_ << "});\n";
}

} // namespace halo