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
    const char* odla_func_name) {
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

  EmitODLACall(ret, odla_func_name, op0, axis.size(), axis, keep_dims,
               EmitShape(ret_type));
  ir_mapping_[*inst] = ret;
}

void GenericCXXCodeGen::RunOnInstruction(ReduceMeanInst* inst) {
  RunOnReductionInstruction(inst, inst->GetAxis(), inst->GetKeepDims(),
                            "odla_ReduceMean");
}

void GenericCXXCodeGen::RunOnInstruction(ReduceMinInst* inst) {
  RunOnReductionInstruction(inst, inst->GetAxis(), inst->GetKeepDims(),
                            "odla_ReduceMin");
}

void GenericCXXCodeGen::RunOnInstruction(ReduceMaxInst* inst) {
  RunOnReductionInstruction(inst, inst->GetAxis(), inst->GetKeepDims(),
                            "odla_ReduceMax");
}

void GenericCXXCodeGen::RunOnInstruction(ReduceL1Inst* inst) {
  RunOnReductionInstruction(inst, inst->GetAxis(), inst->GetKeepDims(),
                            "odla_ReduceL1");
}

void GenericCXXCodeGen::RunOnInstruction(ReduceL2Inst* inst) {
  RunOnReductionInstruction(inst, inst->GetAxis(), inst->GetKeepDims(),
                            "odla_ReduceL2");
}

void GenericCXXCodeGen::RunOnInstruction(ReduceProductInst* inst) {
  RunOnReductionInstruction(inst, inst->GetAxis(), inst->GetKeepDims(),
                            "odla_ReduceProd");
}

void GenericCXXCodeGen::RunOnInstruction(ReduceSumInst* inst) {
  RunOnReductionInstruction(inst, inst->GetAxis(), inst->GetKeepDims(),
                            "odla_ReduceSum");
}

void GenericCXXCodeGen::RunOnInstruction(ReduceLogSumInst* inst) {
  RunOnReductionInstruction(inst, inst->GetAxis(), inst->GetKeepDims(),
                            "odla_ReduceLogSum");
}

void GenericCXXCodeGen::RunOnInstruction(ReduceLogSumExpInst* inst) {
  RunOnReductionInstruction(inst, inst->GetAxis(), inst->GetKeepDims(),
                            "odla_ReduceLogSumExp");
}

void GenericCXXCodeGen::RunOnInstruction(ReduceSumSquareInst* inst) {
  RunOnReductionInstruction(inst, inst->GetAxis(), inst->GetKeepDims(),
                            "odla_ReduceSumSquare");
}

} // namespace halo
