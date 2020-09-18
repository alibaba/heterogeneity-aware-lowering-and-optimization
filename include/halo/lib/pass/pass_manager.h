//===- pass_mgr.h -----------------------------------------------*- C++ -*-===//
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

#ifndef HALO_LIB_PASS_PASS_MGR_H_
#define HALO_LIB_PASS_PASS_MGR_H_

#include <iostream>
#include <list>
#include <memory>

#include "halo/lib/framework/global_context.h"
#include "halo/lib/ir/module.h"
#include "halo/lib/pass/pass.h"

namespace halo {

class FunctionPassManager;
class PassManagerImpl;

// A class that manages all IR transformation passes.
class PassManager final {
 public:
  explicit PassManager(GlobalContext& ctx);
  ~PassManager();

  /// Add a pass to the pass manager.
  template <typename T, typename... TS>
  T* AddPass(TS... args) {
    auto pass = std::make_unique<T>(args...);
    T* ret = static_cast<T*>(pass.get());
    Add(std::move(pass));
    return ret;
  }

  /// Apply all passes on `module`.
  Status Run(Module* module);

  /// Print the pass structure.
  void Print(std::ostream& os) const;

  void Dump() const;

 private:
  Pass* Add(std::unique_ptr<ModulePass> pass);
  Pass* Add(std::unique_ptr<FunctionPass> pass);
  Pass* Add(std::unique_ptr<BasicBlockPass> pass);
  std::unique_ptr<PassManagerImpl> impl_;
};

} // end namespace halo.

#endif // HALO_LIB_PASS_PASS_MGR_H_