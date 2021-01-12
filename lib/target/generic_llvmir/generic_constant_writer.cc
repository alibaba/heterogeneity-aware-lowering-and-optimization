//===- generic_constant_writer.cc -----------------------------------------===//
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

#include "halo/lib/target/generic_llvmir/generic_llvmir_codegen.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/Target/TargetMachine.h"

namespace halo {

GenericConstantWriter::GenericConstantWriter(const std::string& name,
                                             std::ostream& os,
                                             bool bitcode_format)
    : GenericLLVMIRCodeGen(name, ConstantDataStorage::DefinedAsGlobal),
      os_(os),
      bitcode_format_(bitcode_format) {}

GenericConstantWriter::GenericConstantWriter(std::ostream& os,
                                             bool bitcode_format)
    : GenericConstantWriter("Generic Constant Writer", os, bitcode_format) {}

bool GenericConstantWriter::RunOnModule(Module* module) {
  module_ = module;
  if (ctx_ == nullptr) {
    ctx_ = &module->GetGlobalContext();
  }

  if (target_machine_ == nullptr) {
    target_machine_ = InitTargetMachine();
  }
  target_machine_->setOptLevel(llvm::CodeGenOpt::Level::None);
  std::vector<const char*> args;
  HLCHECK(target_machine_);
  llvm_module_ = llvm::make_unique<llvm::Module>(
      module->GetName() + "_constants", GetLLVMContext());
  llvm_module_->setDataLayout(target_machine_->createDataLayout());
  llvm_module_->setTargetTriple(target_machine_->getTargetTriple().getTriple());
  for (auto& c : module->Constants()) {
    RunOnConstant(*c);
  }
  for (auto& func : *module) {
    for (auto& constant : func->Constants()) {
      RunOnConstant(*constant);
    }
  }
  WriteToBuf();
  return false;
}

void GenericConstantWriter::WriteToBuf() {
  llvm::raw_os_ostream llvm_os(os_);
  if (bitcode_format_) {
    llvm::WriteBitcodeToFile(*llvm_module_, llvm_os);
  } else {
    llvm_module_->print(llvm_os, nullptr);
  }
}

ELFConstantWriter::ELFConstantWriter(const std::string& name, std::ostream& os)
    : GenericConstantWriter(name, os, true) {}

ELFConstantWriter::ELFConstantWriter(std::ostream& os)
    : GenericConstantWriter("ELF Constant Writer", os, true) {}

void ELFConstantWriter::WriteToBuf() {
  GlobalContext& ctx = module_->GetGlobalContext();
  llvm::TargetMachine* tm = target_machine_;
  llvm::raw_os_ostream llvm_os(os_);
  llvm::buffer_ostream buf(llvm_os);

  const llvm::Target& target = tm->getTarget();
  const llvm::MCSubtargetInfo& sti = *tm->getMCSubtargetInfo();
  const llvm::MCRegisterInfo& mri = *tm->getMCRegisterInfo();
  std::unique_ptr<llvm::MCAsmBackend> mab(
      target.createMCAsmBackend(sti, mri, tm->Options.MCOptions));

  if (mab->Endian != llvm::support::endian::system_endianness()) {
    // Go through the slow path.
    llvm::legacy::PassManager pm;
    target_machine_->addPassesToEmitFile(
        pm, buf, nullptr,
        llvm::TargetMachine::CodeGenFileType::CGFT_ObjectFile);
    pm.run(*llvm_module_);
    return;
  }

  llvm::LLVMTargetMachine* ltm =
      static_cast<llvm::LLVMTargetMachine*>(tm); // NOLINT
  llvm::MachineModuleInfo mmi(ltm);
  mmi.doInitialization(*llvm_module_);
  llvm::MCContext& mctx = mmi.getContext();
  const llvm::MCInstrInfo& mii = *tm->getMCInstrInfo();
  std::unique_ptr<llvm::MCCodeEmitter> mce(
      target.createMCCodeEmitter(mii, mri, mctx));

  llvm::Triple triple(llvm::Triple::normalize(ctx.GetTargetTriple()));

  std::unique_ptr<llvm::MCStreamer> streamer(target.createMCObjectStreamer(
      triple, mctx, std::move(mab), mab->createObjectWriter(buf),
      std::move(mce), sti, true, true, true));

  auto asm_printer = target.createAsmPrinter(*tm, std::move(streamer));
  llvm::MCStreamer* mc_streamer = asm_printer->OutStreamer.get();
  asm_printer->MMI = &mmi;
  llvm::TargetLoweringObjectFile* objfile_lowering = tm->getObjFileLowering();
  objfile_lowering->Initialize(mctx, *tm);
  mc_streamer->InitSections(false);
  mc_streamer->EmitVersionForTarget(triple, llvm_module_->getSDKVersion());
  asm_printer->EmitStartOfAsmFile(*llvm_module_);

  for (const auto& gv : llvm_module_->globals()) {
    //  AsmPrinter::EmitGlobalVariable(&gv) is slow because it emits each
    //  element one by one.
    llvm::MCSymbol* gv_sym = asm_printer->getSymbol(&gv);
    asm_printer->EmitVisibility(gv_sym, gv.getVisibility(),
                                true /* definition */);
    HLCHECK(gv.hasInitializer());
    mc_streamer->EmitSymbolAttribute(gv_sym, llvm::MCSA_ELF_TypeObject);
    llvm::SectionKind sec_kind =
        llvm::TargetLoweringObjectFile::getKindForGlobal(&gv, *tm);

    const llvm::DataLayout& dl = gv.getParent()->getDataLayout();
    uint64_t size = dl.getTypeAllocSize(gv.getValueType());

    unsigned align = 0;
    if (const llvm::GlobalVariable* gvar =
            llvm::dyn_cast<llvm::GlobalVariable>(&gv)) {
      align = dl.getPreferredAlignment(gvar);
    }
    align = std::max(gv.getAlignment(), align);

    llvm::MCSection* section =
        objfile_lowering->SectionForGlobal(&gv, sec_kind, *tm);

    mc_streamer->SwitchSection(section);

    asm_printer->EmitLinkage(&gv, gv_sym);
    asm_printer->EmitAlignment(llvm::Log2_32(align), &gv);

    mc_streamer->EmitLabel(gv_sym);

    const llvm::Constant* cv = gv.getInitializer();
    const llvm::ConstantDataSequential* cds = nullptr;
    if (cds = llvm::dyn_cast<llvm::ConstantDataSequential>(cv);
        cds != nullptr) {
      size_t bytes = dl.getTypeAllocSize(cds->getType());
      mc_streamer->EmitBytes(cds->getRawDataValues()); // NOLINT
      size_t emitted_bytes =
          dl.getTypeAllocSize(cds->getType()->getElementType()) *
          cds->getNumElements();
      HLCHECK(emitted_bytes == cds->getRawDataValues().size());
      HLCHECK(bytes >= emitted_bytes);
      if (size_t paddings = bytes - emitted_bytes) {
        mc_streamer->EmitZeros(paddings);
      }
    } else {
      asm_printer->EmitGlobalConstant(dl, cv);
    }
    mc_streamer->emitELFSize(gv_sym, llvm::MCConstantExpr::create(size, mctx));
  }
  mc_streamer->Finish();
}

} // namespace halo