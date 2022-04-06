//===- custom.cc ----------------------------------------------------------===//
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

#include "halo/lib/ir/ir_builder.h"
#include "halo/lib/target/generic_cxx/generic_cxx_codegen.h"
#include "halo/lib/transforms/transforms_util.h"

namespace halo {

void GenericCXXCodeGen::RunOnInstruction(CustomInst* inst) {
  std::vector<CXXValue> rets;
  for (int i = 0; i < static_cast<int>(inst->GetNumOfResults()); i++) {
    rets.push_back({inst->GetName() + std::to_string(i),
                    TensorTypeToCXXType(inst->GetResultType(i), false)});
    ir_mapping_[Def(inst, i)] = rets[i];
  }

  std::vector<CXXValue> inputs;
  inputs.reserve(inst->GetNumOfOperands());
  for (auto& op : inst->GetOperands()) {
    inputs.push_back(ir_mapping_[op]);
  }

  if (inst->GetOpName() == "custom_DetectionOutput") {
    const std::string op_name = "\"DetectionOutput\"";
    const int& bl = FindAttributeValue<int>(*inst, "background_label_id");
    float threshold = FindAttributeValue<float>(*inst, "confidence_threshold");
    int keep_top_k = FindAttributeValue<int>(*inst, "keep_top_k");
    int classes = FindAttributeValue<int>(*inst, "num_classes");
    bool share_loc = FindAttributeValue<bool>(*inst, "share_location");
    int code_type = FindAttributeValue<int>(*inst, "code_type");
    float nms_threshold = FindAttributeValue<float>(*inst, "nms_threshold");
    int nms_top_k = FindAttributeValue<int>(*inst, "nms_top_k");
    float nms_eta = FindAttributeValue<float>(*inst, "nms_eta");

    EmitODLACustomCall(rets, inputs, op_name, op_name,
                       ArgWithComment(bl, "background label id"),
                       ArgWithComment(threshold, "conf threshold"),
                       ArgWithComment(keep_top_k, "keep top k"),
                       ArgWithComment(classes, "num classes"),
                       ArgWithComment(share_loc, "share location"),
                       ArgWithComment(code_type, "code type"),
                       ArgWithComment(nms_threshold, "nms threshold"),
                       ArgWithComment(nms_top_k, "nms top k"),
                       ArgWithComment(nms_eta, "nms eta"));
    return;
  }
  const std::string op_name = "\"" + inst->GetOpName() + "\"";
  EmitODLACall(rets, "odla_CustomOp", inputs, op_name, op_name);
}

} // namespace halo