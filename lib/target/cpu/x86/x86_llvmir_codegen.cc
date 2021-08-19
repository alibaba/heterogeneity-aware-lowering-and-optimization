//===- x86_llvmir_codegen.cc ----------------------------------------------===//
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

#include "halo/lib/target/cpu/x86/binary/x86_llvmir_codegen.h"

#include "halo/lib/target/codegen_object.h"
#include "halo/lib/target/generic_llvmir/generic_llvmir_codegen.h"
#include "llvm-c/Target.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/Target/TargetMachine.h"

namespace halo {

static llvm::TargetMachine* GetX86TargetMachine(const GlobalContext& ctx) {
  static llvm::TargetMachine* tm = nullptr;
  if (tm != nullptr) {
    return tm;
  }
  // LLVMInitializeNativeTarget();
  LLVMInitializeX86TargetInfo();
  LLVMInitializeX86Target();
  LLVMInitializeX86TargetMC();
  LLVMInitializeX86AsmPrinter();
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
    cpu = llvm::sys::getHostCPUName();
  }
  llvm::StringRef features(
      ""); // TODO (unknown) : get target specific feature string from ctx.
  llvm::Reloc::Model reloc = llvm::Reloc::Static;
  llvm::CodeGenOpt::Level opt_level = llvm::CodeGenOpt::Aggressive;
  llvm::CodeModel::Model cm = llvm::CodeModel::Medium;
  llvm::TargetOptions options;
  options.UnsafeFPMath = true;
  options.NoInfsFPMath = true;
  options.NoNaNsFPMath = true;
  options.NoSignedZerosFPMath = true;

  tm = target->createTargetMachine(triple.getTriple(), cpu, features, options,
                                   reloc, cm, opt_level);
  return tm;
}

X86LLVMIRCodeGen::X86LLVMIRCodeGen(ConstantDataStorage constant_data_storage)
    : GenericLLVMIRCodeGen("X86 LLVMIR Code Gen", constant_data_storage) {}

X86LLVMIRCodeGen::X86LLVMIRCodeGen()
    : X86LLVMIRCodeGen(ConstantDataStorage::DefinedAsStatic) {}

llvm::TargetMachine* X86LLVMIRCodeGen::InitTargetMachine() {
  return GetX86TargetMachine(*ctx_);
}

X86BinaryWriter::X86BinaryWriter(std::ostream& os)
    : CodeWriter("X86 Binary Writer", os) {}

bool X86BinaryWriter::RunOnModule(Module* module) {
  llvm::legacy::PassManager pm;
  std::error_code ec;
  llvm::raw_os_ostream llvm_os(os_);
  llvm::buffer_ostream buf(llvm_os);
  GlobalContext& ctx = module->GetGlobalContext();

  auto llvm_module =
      module->GetGlobalContext().GetCodeGenObject().GetLLVMModule();

  GetX86TargetMachine(ctx)->addPassesToEmitFile(
      pm, buf, nullptr, llvm::CodeGenFileType::CGFT_ObjectFile);
  pm.run(*llvm_module);
  return false;
}

X86ConstantWriter::X86ConstantWriter(std::ostream& os)
    : ELFConstantWriter("X86 Constant Writer", os) {}

llvm::TargetMachine* X86ConstantWriter::InitTargetMachine() {
  return GetX86TargetMachine(*ctx_);
}

} // namespace halo