//===- codegen_object.cc --------------------------------------------------===//
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

#include "halo/lib/target/codegen_object.h"

#include "llvm/IR/Module.h"

namespace halo {

CodeGenObject::CodeGenObject() : llvm_module_(nullptr) {}

CodeGenObject::~CodeGenObject() = default;

llvm::Module* CodeGenObject::GetLLVMModule() const noexcept {
  return llvm_module_.get();
}

void CodeGenObject::SetLLVMModule(std::unique_ptr<llvm::Module> module) {
  llvm_module_ = std::move(module);
}

} // namespace halo