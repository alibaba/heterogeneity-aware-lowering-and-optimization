//===- function.cc --------------------------------------------------------===//
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

#include "halo/lib/ir/function.h"

#include "halo/lib/ir/common_instructions.h"

namespace halo {

ReturnInst* Function::GetReturnInst() const {
  HLCHECK(!basic_blocks_.empty());
  for (auto& bb : basic_blocks_) {
    // Skip bbs for loop body (with arguments)
    if (bb->Args().empty()) {
      auto ret = bb->GetReturnInst();
      if (ret != nullptr) {
        return ret;
      }
    }
  }
  return nullptr;
}

void Function::Print(std::ostream& os) const {
  os << "Function: " << GetName() << "(";
  int arg_idx = 0;
  for (auto& arg : Args()) {
    if (arg_idx++ != 0) {
      os << ", ";
    }
    arg->Print(os);
  }
  os << ")\n";

  for (auto& c : Constants()) {
    c->Print(os);
  }

  for (auto& bb : *this) {
    bb->Print(os);
  }
}

} // namespace halo
