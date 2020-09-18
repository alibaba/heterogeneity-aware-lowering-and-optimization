//===- nms.cc -------------------------------------------------------------===//
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

#include "halo/lib/target/generic_cxx/generic_cxx_codegen.h"

namespace halo {

void GenericCXXCodeGen::RunOnInstruction(NonMaxSuppressionInst* inst) {
  const Def& boxes = inst->GetOperand(0);
  const Def& scores = inst->GetOperand(1);

  CXXValue op0 = ir_mapping_[boxes];
  CXXValue op1 = ir_mapping_[scores];

  uint32_t max_num_outputs = 0;
  float iou_threshold = 0.0;
  float score_threshold = 0.0;
  if (inst->GetNumOfOperands() > 4) {
    const auto& op2 = inst->GetOperand(2);
    HLCHECK(IsA<Constant>(op2));
    const Constant* c_k = DynCast<Constant>(op2);
    const auto& op2_type = op2.GetType();
    HLCHECK(op2_type.GetTotalNumOfElements() == 1);
    if (op2_type.GetDataType() == DataType::INT32) {
      max_num_outputs = static_cast<uint32_t>(c_k->GetData<int32_t>(0));
    } else if (op2_type.GetDataType() == DataType::INT64) {
      max_num_outputs = static_cast<uint32_t>(c_k->GetData<int64_t>(0));
    }

    const auto& op3 = inst->GetOperand(3);
    HLCHECK(IsA<Constant>(op3));
    const Constant* c_iou = DynCast<Constant>(op3);
    const auto& op3_type = op3.GetType();
    HLCHECK(op3_type.GetTotalNumOfElements() == 1);
    if (op3_type.GetDataType() == DataType::FLOAT32) {
      iou_threshold = c_iou->GetData<float>(0);
    }

    const auto& op4 = inst->GetOperand(4);
    HLCHECK(IsA<Constant>(op4));
    const Constant* c_score = DynCast<Constant>(op4);
    const auto& op4_type = op4.GetType();
    HLCHECK(op4_type.GetTotalNumOfElements() == 1);
    if (op4_type.GetDataType() == DataType::FLOAT32) {
      score_threshold = c_score->GetData<float>(0);
    }
  }

  CXXValue ret(inst->GetName(), op0.type);
  const auto& ret_type = inst->GetResultType();

  EmitODLACall(ret, "odla_NMS", op0, op1, max_num_outputs, iou_threshold,
               score_threshold, ret_type);
  ir_mapping_[*inst] = ret;
}

} // end namespace halo