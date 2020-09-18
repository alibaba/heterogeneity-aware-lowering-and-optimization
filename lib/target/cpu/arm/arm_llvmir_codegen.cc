//===- arm_llvmir_codegen.cc ----------------------------------------------===//
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

#include "halo/lib/target/cpu/arm/binary/arm_llvmir_codegen.h"

#include "llvm-c/Target.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "halo/lib/target/codegen_object.h"
#include "halo/lib/target/generic_llvmir/generic_llvmir_codegen.h"

namespace halo {

static llvm::TargetMachine* GetARMTargetMachine(const GlobalContext& ctx) {
  static llvm::TargetMachine* tm = nullptr;
  if (tm != nullptr) {
    return tm;
  }
  LLVMInitializeAArch64TargetInfo();
  LLVMInitializeAArch64Target();
  LLVMInitializeAArch64TargetMC();
  LLVMInitializeAArch64AsmPrinter();
  llvm::Triple triple(llvm::Triple::normalize(ctx.GetTargetTriple()));
  std::string error;
  const llvm::Target* target =
      llvm::TargetRegistry::lookupTarget(triple.getTriple(), error);
  if (target == nullptr) {
    llvm::errs() << error;
    return nullptr;
  }
  HLCHECK(target);
  auto cpu = ctx.GetProcessorName();
  if (cpu.empty() || cpu == "native") {
    cpu = "cortex-a57";
  }
  llvm::StringRef features("+fp-armv8");
  llvm::Reloc::Model reloc = llvm::Reloc::Static;
  llvm::CodeGenOpt::Level opt_level = llvm::CodeGenOpt::Aggressive;
  llvm::CodeModel::Model cm = llvm::CodeModel::Large;
  llvm::TargetOptions options;
  options.UnsafeFPMath = true;
  options.NoInfsFPMath = true;
  options.NoNaNsFPMath = true;
  options.NoSignedZerosFPMath = true;

  tm = target->createTargetMachine(triple.getTriple(), cpu, features, options,
                                   reloc, cm, opt_level);
  return tm;
}

ARMLLVMIRCodeGen::ARMLLVMIRCodeGen(ConstantDataStorage constant_data_storage)
    : GenericLLVMIRCodeGen("ARM LLVMIR Code Gen", constant_data_storage) {}

ARMLLVMIRCodeGen::ARMLLVMIRCodeGen()
    : ARMLLVMIRCodeGen(ConstantDataStorage::DefinedAsStatic) {}

llvm::TargetMachine* ARMLLVMIRCodeGen::InitTargetMachine() {
  return GetARMTargetMachine(*ctx_);
}

ARMBinaryWriter::ARMBinaryWriter(std::ostream& os)
    : CodeWriter("ARM Binary Writer", os) {}

bool ARMBinaryWriter::RunOnModule(Module* module) {
  llvm::legacy::PassManager pm;
  std::error_code ec;
  llvm::raw_os_ostream llvm_os(os_);
  llvm::buffer_ostream buf(llvm_os);
  GlobalContext& ctx = module->GetGlobalContext();

  auto llvm_module =
      module->GetGlobalContext().GetCodeGenObject().GetLLVMModule();

  GetARMTargetMachine(ctx)->addPassesToEmitFile(
      pm, buf, nullptr, llvm::TargetMachine::CodeGenFileType::CGFT_ObjectFile);
  pm.run(*llvm_module);
  return false;
}

ARMConstantWriter::ARMConstantWriter(std::ostream& os)
    : ELFConstantWriter("ARM Constant Writer", os) {}

llvm::TargetMachine* ARMConstantWriter::InitTargetMachine() {
  return GetARMTargetMachine(*ctx_);
}

} // namespace halo