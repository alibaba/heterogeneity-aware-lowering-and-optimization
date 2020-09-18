//===- codegen.cc ---------------------------------------------------------===//
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

#include "halo/lib/target/codegen.h"

#include "halo/lib/ir/all_instructions.h"

namespace halo {

void CodeGen::RunOnBaseInstruction(Instruction* inst) {
  switch (inst->GetOpCode()) {
#define GET_INST_DOWNCAST_SWITCH
#include "halo/lib/ir/instructions_info.def"
#undef GET_INST_DOWNCAST_SWITCH
    default: {
      HLCHECK(0 && "Unreachable");
    }
  }
}

const std::string& CodeGen::GetRTLibFuncName(const Instruction& inst) {
  auto kv = RuntimeLibFuncNames.find(inst.GetOpCode());
  HLCHECK(kv != RuntimeLibFuncNames.end());
  return kv->second;
}

} // namespace halo