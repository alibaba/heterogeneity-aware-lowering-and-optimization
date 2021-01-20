//===- topk.cc ------------------------------------------------------------===//
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

void GenericCXXCodeGen::RunOnInstruction(TopKInst* inst) {
  const Def& input = inst->GetOperand(0);

  CXXValue op0 = ir_mapping_[input];
  const auto& ret_type = inst->GetResultType();

  const Def& topk = inst->GetOperand(1);
  const static uint32_t max_topk_num = 25;
  uint32_t k = max_topk_num;
  if (IsA<Constant>(topk)) {
    const Constant* c_k = DynCast<Constant>(topk);
    const auto& op1_type = topk.GetType();
    HLCHECK(op1_type.GetTotalNumOfElements() == 1);
    if (op1_type.GetDataType() == DataType::INT32) {
      k = static_cast<uint32_t>(c_k->GetData<int32_t>(0));
    } else if (op1_type.GetDataType() == DataType::INT64) {
      k = static_cast<uint32_t>(c_k->GetData<int64_t>(0));
    }
  }
  std::vector<CXXValue> rets;
  rets.emplace_back(inst->GetName(), op0.type);
  rets.emplace_back(inst->GetName() + "_index",
                    TensorTypeToCXXType(inst->GetResultType(1), false));

  const auto& axis = inst->GetAxis();
  const auto& largest = inst->GetLargest();
  const auto& sorted = inst->GetSorted();

  EmitODLACall(rets, "odla_TopK", op0, k, largest, sorted, axis, ret_type,
               inst->GetResultType(1));
  ir_mapping_[*inst] = rets[0];
  ir_mapping_[Def(inst, 1)] = rets[1];
}

} // namespace halo