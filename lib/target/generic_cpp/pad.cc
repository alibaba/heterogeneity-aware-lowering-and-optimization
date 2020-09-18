//===- pad.cc -------------------------------------------------------------===//
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

#include <cstdio>
#include <string>

#include "halo/lib/ir/common_reduction_instructions.h"
#include "halo/lib/ir/constant.h"
#include "halo/lib/ir/ir_builder.h"
#include "halo/lib/ir/nn_cnn_instructions.h"
#include "halo/lib/ir/nn_instructions.h"
#include "halo/lib/target/generic_cxx/generic_cxx_codegen.h"

namespace halo {

void GenericCXXCodeGen::RunOnInstruction(PadInst* inst) {
  const Def& lhs = inst->GetOperand(0);
  const Def& rhs = inst->GetOperand(1);
  const auto& lhs_type = lhs.GetType();

  HLCHECK(IsA<Constant>(rhs) && inst->GetMode() == PadMode::CONSTANT);

  const Constant* pad_amt = DynCast<Constant>(rhs.GetOwner());
  size_t dims = lhs_type.GetNumOfDims();
  HLCHECK(pad_amt->GetResultType().GetDataType() == DataType::INT32 &&
          (size_t)pad_amt->GetResultType().GetTotalNumOfElements() == dims * 2);
  std::vector<uint32_t> paddings_front(dims);
  std::vector<uint32_t> paddings_back(dims);
  for (size_t i = 0; i < dims; ++i) {
    paddings_front[i] = pad_amt->GetData<int32_t>(i * 2);
    paddings_back[i] = pad_amt->GetData<int32_t>(i * 2 + 1);
  }

  CXXValue op0 = ir_mapping_[lhs];
  CXXValue ret(inst->GetName(), op0.type);
  EmitODLACall(ret, "odla_Pad", op0, paddings_front, paddings_back,
               EmitShape(inst->GetResultType()));
  ir_mapping_[*inst] = ret;
}

} // namespace halo