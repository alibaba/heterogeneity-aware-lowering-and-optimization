//===- riscv_llvmir_codegen.cc --------------------------------------------===//
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

#include "halo/lib/target/cpu/riscv/binary/riscv_llvmir_codegen.h"

#include "llvm-c/Target.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/MC/MCTargetOptions.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "halo/lib/target/codegen_object.h"
#include "halo/lib/target/generic_llvmir/generic_llvmir_codegen.h"

namespace halo {

static llvm::TargetMachine* GetRISCVTargetMachine(const GlobalContext& ctx) {
  static llvm::TargetMachine* tm = nullptr;
  if (tm != nullptr) {
    return tm;
  }
  LLVMInitializeRISCVTargetInfo();
  LLVMInitializeRISCVTarget();
  LLVMInitializeRISCVTargetMC();
  LLVMInitializeRISCVAsmPrinter();
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
    cpu = "generic-rv32";
  }

  // TODO (unknown) : get target specific feature string from ctx.
  llvm::StringRef features("+a,+m,+f,+d,+c,+relax");
  llvm::Reloc::Model reloc = llvm::Reloc::Static;
  llvm::CodeGenOpt::Level opt_level = llvm::CodeGenOpt::Aggressive;
  llvm::CodeModel::Model cm = llvm::CodeModel::Medium;
  llvm::TargetOptions options;
  options.ExceptionModel = llvm::ExceptionHandling::None;
  options.MCOptions.ABIName = "ilp32d";
  options.UnsafeFPMath = true;
  options.NoInfsFPMath = true;
  options.NoNaNsFPMath = true;
  options.NoSignedZerosFPMath = true;

  tm = target->createTargetMachine(triple.getTriple(), cpu, features, options,
                                   reloc, cm, opt_level);
  return tm;
}

RISCVLLVMIRCodeGen::RISCVLLVMIRCodeGen(
    ConstantDataStorage constant_data_storage, std::string rt_lib_name)
    : GenericLLVMIRCodeGen("RISCV LLVMIR Code Gen", constant_data_storage) {
  RuntimeLibName = std::move(rt_lib_name);
}

RISCVLLVMIRCodeGen::RISCVLLVMIRCodeGen()
    : RISCVLLVMIRCodeGen(ConstantDataStorage::DefinedAsStatic) {}

llvm::TargetMachine* RISCVLLVMIRCodeGen::InitTargetMachine() {
  return GetRISCVTargetMachine(*ctx_);
}

RISCVBinaryWriter::RISCVBinaryWriter(std::ostream& os)
    : CodeWriter("RISCV Binary Writer", os) {}

bool RISCVBinaryWriter::RunOnModule(Module* module) {
  llvm::legacy::PassManager pm;
  std::error_code ec;
  llvm::raw_os_ostream llvm_os(os_);
  llvm::buffer_ostream buf(llvm_os);
  GlobalContext& ctx = module->GetGlobalContext();

  auto llvm_module =
      module->GetGlobalContext().GetCodeGenObject().GetLLVMModule();

  GetRISCVTargetMachine(ctx)->addPassesToEmitFile(
      pm, buf, nullptr, llvm::TargetMachine::CodeGenFileType::CGFT_ObjectFile);
  pm.run(*llvm_module);
  return false;
}

RISCVConstantWriter::RISCVConstantWriter(std::ostream& os)
    : ELFConstantWriter("RISCV Constant Writer", os) {}

llvm::TargetMachine* RISCVConstantWriter::InitTargetMachine() {
  return GetRISCVTargetMachine(*ctx_);
}

} // namespace halo