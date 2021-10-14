//===- quantization.cc ----------------------------------------------------===//
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

#include "halo/lib/ir/common_instructions.h"
#include "halo/lib/ir/quantization_instructions.h"
#include "halo/lib/target/generic_cxx/generic_cxx_codegen.h"

namespace halo {

void GenericCXXCodeGen::RunOnInstruction(DequantizeInst* inst) {
  const auto& axis = inst->GetAxis();

  CXXValue op0 = ir_mapping_[inst->GetOperand(0)];
  CXXValue op_scale = ir_mapping_[inst->GetOperand(1)];
  CXXValue op_zp = inst->GetNumOfOperands() == 3
                       ? ir_mapping_[inst->GetOperand(2)]
                       : ir_mapping_[Def::GetUndefined()];

  CXXValue ret(inst->GetName(), op0.type);

  ir_mapping_[*inst] = ret;
  EmitODLACall(ret, "odla_Dequantize", op0, op_scale, op_zp, axis,
               DataType::FLOAT32);
}

void GenericCXXCodeGen::RunOnInstruction(QuantizeInst* inst) {
  const auto& axis = inst->GetAxis();

  CXXValue op0 = ir_mapping_[inst->GetOperand(0)];
  const auto undef = ir_mapping_[Def::GetUndefined()];
  CXXValue op_scale =
      inst->GetNumOfOperands() > 1 ? ir_mapping_[inst->GetOperand(1)] : undef;
  CXXValue op_zp =
      inst->GetNumOfOperands() == 3 ? ir_mapping_[inst->GetOperand(2)] : undef;

  CXXValue ret(inst->GetName(), op0.type);

  ir_mapping_[*inst] = ret;
  EmitODLACall(ret, "odla_Quantize", op0, op_scale, op_zp, axis,
               inst->GetResultsTypes()[0].GetDataType());
  if (inst->GetNumOfOperands() == 1) {
  }
}

} // namespace halo
