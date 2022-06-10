//===- return.cc ----------------------------------------------------------===//
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

#include "halo/lib/ir/ir_builder.h"
#include "halo/lib/target/generic_cxx/generic_cxx_codegen.h"

namespace halo {

void GenericCXXCodeGen::RunOnInstruction(ReturnInst* inst) {
  bool is_compile_mode = opts_.exec_mode == ExecMode::Compile;
  for (auto& op : inst->GetOperands()) {
    const CXXValue& val = ir_mapping_[op];
    if (is_compile_mode) {
      bool is_entry_with_calls =
          inst->GetParent()->GetParent()->IsEntryFunction() &&
          inst->GetParent()->GetParent()->GetParent()->Functions().size() > 1;
      if (!is_entry_with_calls) {
        os_ << "  odla_SetValueAsOutput(" << val.name << ");\n";
      }
    } else {
      os_ << "  odla_GetValueData("
          << Join(val.name, "out_" + val.name, EmitNull()) << ");\n";
    }
  }
}

} // namespace halo