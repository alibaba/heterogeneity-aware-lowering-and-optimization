//===- generic_llvmir_codegen.h ---------------------------------*- C++ -*-===//
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

#ifndef HALO_LIB_TARGET_CODEGEN_OBJECT_H_
#define HALO_LIB_TARGET_CODEGEN_OBJECT_H_

#include <memory>

namespace llvm {
class Module;
}

namespace halo {

// The class holds the generated code and related context information.
class CodeGenObject final {
 public:
  CodeGenObject();
  ~CodeGenObject();

  llvm::Module* GetLLVMModule() const noexcept;
  void SetLLVMModule(std::unique_ptr<llvm::Module> module);

 private:
  std::unique_ptr<llvm::Module> llvm_module_;
};

} // end namespace halo.

#endif // HALO_LIB_TARGET_CODEGEN_TARGET_H_