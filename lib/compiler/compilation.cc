//===- compilation.cc -----------------------------------------------------===//
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

#include "halo/lib/compiler/compilation.h"

#include "halo/lib/framework/global_context.h"
#include "halo/lib/ir/module.h"

namespace halo {

/// Constructor of the Compilation class.
Compilation::Compilation(GlobalContext& context) {
  // Give a default module name "halo_main"
  const std::string module_name = "halo_main";
  module_ = std::make_unique<Module>(context, module_name);
}

Compilation::~Compilation() {}

} // namespace halo