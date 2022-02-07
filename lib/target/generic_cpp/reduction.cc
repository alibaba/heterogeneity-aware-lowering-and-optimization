//===- reduction.cc -------------------------------------------------------===//
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

void GenericCXXCodeGen::RunOnReductionInstruction(
    Instruction* inst, const std::vector<int32_t>& axis_attr, bool keep_dims,
    const std::string& odla_func_name, float eps) {
  const Def& input = inst->GetOperand(0);

  CXXValue op0 = ir_mapping_[input];
  const auto& ret_type = inst->GetResultType();

  CXXValue ret(inst->GetName(), op0.type);
  std::vector<uint32_t> axis;

  size_t dims = input.GetType().GetNumOfDims();
  axis.reserve(dims);
  for (auto x : axis_attr) {
    axis.push_back(x);
  }
  if (axis.empty()) {
    for (size_t i = 0; i < dims; ++i) {
      axis.push_back(i);
    }
  }
  if (odla_func_name == "odla_ReduceL1" || odla_func_name == "odla_ReduceL2") {
    EmitODLACall(ret, odla_func_name.c_str(), op0, axis.size(), axis, keep_dims,
                 eps, EmitShape(ret_type));
  } else {
    EmitODLACall(ret, odla_func_name.c_str(), op0, axis.size(), axis, keep_dims,
                 EmitShape(ret_type));
  }
  ir_mapping_[*inst] = ret;
}

void GenericCXXCodeGen::RunOnInstruction(ReduceMeanInst* inst) {
  RunOnReductionInstruction(inst, inst->GetAxis(), inst->GetKeepDims(),
                            "odla_ReduceMean", 0);
}

void GenericCXXCodeGen::RunOnInstruction(ReduceMinInst* inst) {
  RunOnReductionInstruction(inst, inst->GetAxis(), inst->GetKeepDims(),
                            "odla_ReduceMin", 0);
}

void GenericCXXCodeGen::RunOnInstruction(ReduceMaxInst* inst) {
  RunOnReductionInstruction(inst, inst->GetAxis(), inst->GetKeepDims(),
                            "odla_ReduceMax", 0);
}

void GenericCXXCodeGen::RunOnInstruction(ReduceL1Inst* inst) {
  RunOnReductionInstruction(inst, inst->GetAxis(), inst->GetKeepDims(),
                            "odla_ReduceL1", inst->GetEpsilon());
}

void GenericCXXCodeGen::RunOnInstruction(ReduceL2Inst* inst) {
  RunOnReductionInstruction(inst, inst->GetAxis(), inst->GetKeepDims(),
                            "odla_ReduceL2", inst->GetEpsilon());
}

void GenericCXXCodeGen::RunOnInstruction(ReduceProductInst* inst) {
  RunOnReductionInstruction(inst, inst->GetAxis(), inst->GetKeepDims(),
                            "odla_ReduceProd", 0);
}

void GenericCXXCodeGen::RunOnInstruction(ReduceSumInst* inst) {
  RunOnReductionInstruction(inst, inst->GetAxis(), inst->GetKeepDims(),
                            "odla_ReduceSum", 0);
}

void GenericCXXCodeGen::RunOnInstruction(ReduceLogSumInst* inst) {
  RunOnReductionInstruction(inst, inst->GetAxis(), inst->GetKeepDims(),
                            "odla_ReduceLogSum", 0);
}

void GenericCXXCodeGen::RunOnInstruction(ReduceLogSumExpInst* inst) {
  RunOnReductionInstruction(inst, inst->GetAxis(), inst->GetKeepDims(),
                            "odla_ReduceLogSumExp", 0);
}

void GenericCXXCodeGen::RunOnInstruction(ReduceSumSquareInst* inst) {
  RunOnReductionInstruction(inst, inst->GetAxis(), inst->GetKeepDims(),
                            "odla_ReduceSumSquare", 0);
}

void GenericCXXCodeGen::RunOnInstruction(CumSumInst* inst) {
  const Def& input = inst->GetOperand(0);
  const Def& dims = inst->GetOperand(1);

  CXXValue op0 = ir_mapping_[input];
  CXXValue op1 = ir_mapping_[dims];

  CXXValue ret(inst->GetName(), op0.type);
  EmitODLACall(ret, "odla_CumSum", op0, op1, inst->GetExclusive(),
               inst->GetReverse());
  ir_mapping_[*inst] = ret;
}

} // namespace halo
