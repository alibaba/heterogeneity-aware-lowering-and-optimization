//===- arm_llvmir_codegen.h -------------------------------------*- C++ -*-===//
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

#ifndef HALO_LIB_TARGET_CPU_ARM_ARM_LLVMIR_CODEGEN_H_
#define HALO_LIB_TARGET_CPU_ARM_ARM_LLVMIR_CODEGEN_H_

#include "halo/lib/target/generic_llvmir/generic_llvmir_codegen.h"

namespace halo {

// The ARM LLVMIR code generator.
class ARMLLVMIRCodeGen final : public GenericLLVMIRCodeGen {
 public:
  ARMLLVMIRCodeGen(ConstantDataStorage constant_data_storage);
  ARMLLVMIRCodeGen();

 protected:
  virtual llvm::TargetMachine* InitTargetMachine() override;
};

// The binary writer for ARM target.
class ARMBinaryWriter final : public CodeWriter {
 public:
  explicit ARMBinaryWriter(std::ostream& os);

  bool RunOnModule(Module* module) override;
};

// The constant writer for ARM target.
class ARMConstantWriter final : public ELFConstantWriter {
 public:
  explicit ARMConstantWriter(std::ostream& os);

 protected:
  llvm::TargetMachine* InitTargetMachine() override;
};

} // end namespace halo.

#endif // HALO_LIB_TARGET_CPU_ARM_ARM_LLVMIR_CODEGEN_H_