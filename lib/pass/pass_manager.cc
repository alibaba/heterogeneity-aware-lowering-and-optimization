//===- pass_manager.cc ----------------------------------------------------===//
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

#include "halo/lib/pass/pass_manager.h"

#include <ostream>

#include "halo/lib/quantizer/weights_quantizer.h"
#include "halo/lib/serializer/serializer.h"
#include "halo/lib/target/cpu/arm/binary/arm_llvmir_codegen.h"
#include "halo/lib/target/cpu/riscv/binary/riscv_llvmir_codegen.h"
#include "halo/lib/target/cpu/x86/binary/x86_llvmir_codegen.h"
#include "halo/lib/target/generic_cxx/generic_cxx_codegen.h"
#include "halo/lib/target/generic_cxx/templated_cxx_codegen.h"
#include "halo/lib/target/generic_llvmir/generic_llvmir_codegen.h"
#include "halo/lib/target/jit_compiler/cxx_jit_compiler.h"
#include "halo/lib/target/jit_compiler/generic_jit_linker.h"
#include "halo/lib/target/triton/triton_config_writer.h"
#include "halo/lib/transforms/analyzer.h"
#include "halo/lib/transforms/caffeextension_legalizer.h"
#include "halo/lib/transforms/constant_decombine.h"
#include "halo/lib/transforms/convert_tf_cfg.h"
#include "halo/lib/transforms/dce.h"
#include "halo/lib/transforms/device_placement.h"
#include "halo/lib/transforms/fusion.h"
#include "halo/lib/transforms/input_legalizer.h"
#include "halo/lib/transforms/input_rewriter.h"
#include "halo/lib/transforms/inst_simplify.h"
#include "halo/lib/transforms/onnxextension_legalizer.h"
#include "halo/lib/transforms/output_rewriter.h"
#include "halo/lib/transforms/reorder_channel.h"
#include "halo/lib/transforms/splitting.h"
#include "halo/lib/transforms/tfextension_legalizer.h"
#include "halo/lib/transforms/tfliteextension_legalizer.h"
#include "halo/lib/transforms/type_legalizer.h"
#include "halo/lib/transforms/typecast.h"

static bool IsPrintPass = false;

namespace halo {

class PassManagerImpl {
 public:
  explicit PassManagerImpl(GlobalContext* ctx) : ctx_(*ctx) {
    IsPrintPass = ctx->GetPrintPass();
  }

  Pass* Add(std::unique_ptr<ModulePass> pass);
  Pass* Add(std::unique_ptr<FunctionPass> pass);
  Pass* Add(std::unique_ptr<BasicBlockPass> pass);

  GlobalContext& GetGlobalContext() { return ctx_; }

  Status Run(Module* module);

  void Print(std::ostream& os) const;

 private:
  FunctionPassManager* GetFunctionPassManager();

  GlobalContext& ctx_;
  std::list<std::unique_ptr<ModulePass>> passes_;
}; // namespace halo

PassManager::PassManager(GlobalContext& ctx)
    : impl_(std::make_unique<PassManagerImpl>(&ctx)) {}

PassManager::~PassManager() {}

Pass* PassManager::Add(std::unique_ptr<ModulePass> pass) {
  return impl_->Add(std::move(pass));
}

Pass* PassManager::Add(std::unique_ptr<FunctionPass> pass) {
  return impl_->Add(std::move(pass));
}

Pass* PassManager::Add(std::unique_ptr<BasicBlockPass> pass) {
  return impl_->Add(std::move(pass));
}

GlobalContext& PassManager::GetGlobalContext() const {
  return impl_->GetGlobalContext();
}

Status PassManager::Run(Module* module) { return impl_->Run(module); }

void PassManager::Print(std::ostream& os) const { impl_->Print(os); }

void PassManager::Dump() const { Print(GlobalContext::Dbgs()); }

// BasicBlockPassManager is a function level pass that contains basic block
// passes.
class BasicBlockPassManager final : public FunctionPass {
 public:
  BasicBlockPassManager() : FunctionPass("BasicBlockPassManager") {}
  bool RunOnFunction(Function* function) override {
    bool global_changed = false;
    for (bool changed = true; changed;) {
      changed = false;
      for (auto& bb : *function) {
        for (auto& fp : passes_) {
          if (IsPrintPass) {
            std::cout << "      BasicBlockPass : " << fp->Name() << std::endl;
          }
          changed |= fp->RunOnBasicBlock(bb.get());
          global_changed |= changed;
          if (IsPrintPass && global_changed) {
            std::cout << " ---- After " << fp->Name() << std::endl;
            bb->Dump();
          }
        }
      }
    }
    return global_changed;
  }
  void AddPass(std::unique_ptr<BasicBlockPass> pass) {
    passes_.push_back(std::move(pass));
  }

  void Print(std::ostream& os) const override {
    os << Name() << "\n";
    for (auto& pass : passes_) {
      pass->Print(os);
    }
  }

  bool IsPassManager() const noexcept override { return true; }

 private:
  std::list<std::unique_ptr<BasicBlockPass>> passes_;
};

// FunctionPassManager is a module level pass that contains function passes.
class FunctionPassManager final : public ModulePass {
 public:
  FunctionPassManager() : ModulePass("FunctionPassManager") {}
  bool RunOnModule(Module* module) override {
    bool global_changed = false;
    for (bool changed = true; changed;) {
      changed = false;
      for (auto& func : *module) {
        for (auto& fp : passes_) {
          if (IsPrintPass) {
            std::cout << "    FunctionPass : " << fp->Name() << std::endl;
          }
          changed |= fp->RunOnFunction(func.get());
          global_changed |= changed;
          if (IsPrintPass && global_changed) {
            std::cout << " ---- After " << fp->Name() << std::endl;
            func->Dump();
          }
        }
      }
    }
    return global_changed;
  }

  void AddPass(std::unique_ptr<FunctionPass> pass) {
    passes_.push_back(std::move(pass));
  }

  BasicBlockPassManager* GetBasicBlockPassManager() {
    if (passes_.empty() || !passes_.back()->IsPassManager()) {
      passes_.push_back(std::make_unique<BasicBlockPassManager>());
    }
    BasicBlockPassManager* fpm =
        Downcast<BasicBlockPassManager>(passes_.back().get());
    return fpm;
  }

  void Print(std::ostream& os) const override {
    os << Name() << "\n";
    for (auto& pass : passes_) {
      pass->Print(os);
    }
  }

  bool IsPassManager() const noexcept override { return true; }

 private:
  std::list<std::unique_ptr<FunctionPass>> passes_;
};

Pass* PassManagerImpl::Add(std::unique_ptr<ModulePass> pass) {
  passes_.push_back(std::move(pass));
  return passes_.back().get();
}

Pass* PassManagerImpl::Add(std::unique_ptr<FunctionPass> pass) {
  Pass* ret = pass.get();
  FunctionPassManager* fpm = GetFunctionPassManager();
  fpm->AddPass(std::move(pass));
  return ret;
}

Pass* PassManagerImpl::Add(std::unique_ptr<BasicBlockPass> pass) {
  Pass* ret = pass.get();
  FunctionPassManager* fpm = GetFunctionPassManager();
  BasicBlockPassManager* bpm = fpm->GetBasicBlockPassManager();
  bpm->AddPass(std::move(pass));
  return ret;
}

Status PassManagerImpl::Run(Module* module) {
  for (auto& pass : passes_) {
    if (IsPrintPass) {
      std::cout << "  ModulePass : " << pass->Name() << std::endl;
    }
    pass->RunOnModule(module);
    if (IsPrintPass) {
      std::cout << " ---- After " << pass->Name() << std::endl;
      module->Dump();
    }
  }
  return Status::SUCCESS;
}

FunctionPassManager* PassManagerImpl::GetFunctionPassManager() {
  if (passes_.empty() || !passes_.back()->IsPassManager()) {
    passes_.push_back(std::make_unique<FunctionPassManager>());
  }
  FunctionPassManager* fpm =
      Downcast<FunctionPassManager>(passes_.back().get());
  return fpm;
}

void PassManagerImpl::Print(std::ostream& os) const {
  for (auto& pass : passes_) {
    pass->Print(os);
  }
}

Pass* PassManager::AddAnalyzerPass(std::ostream* os, const AnalyzerOpts& opts) {
  return AddPass<Analyzer>(os, opts);
}

Pass* PassManager::AddARMBinaryWriterPass(std::ostream& os) {
  return AddPass<ARMBinaryWriter>(os);
}

Pass* PassManager::AddARMConstantWriterPass(std::ostream& os) {
  return AddPass<ARMConstantWriter>(os);
}

Pass* PassManager::AddARMLLVMIRCodeGenPass(
    ConstantDataStorage constant_data_storage) {
  return AddPass<ARMLLVMIRCodeGen>(constant_data_storage);
}

Pass* PassManager::AddCAFFEExtensionLegalizerPass() {
  return AddPass<CAFFEExtensionLegalizer>();
}

Pass* PassManager::AddCodeFormatterPass(std::ostringstream& buf_code,
                                        std::ostringstream& buf_header,
                                        const CXXCodeGenOpts& opts) {
  return AddPass<CodeFormatter>(buf_code, buf_header, opts);
}

Pass* PassManager::AddConstantWriterPass(std::ostream& os,
                                         const std::string& target) {
  auto is_begin_with = [](const std::string& s, const std::string& t) {
    return s.substr(0, t.size()) == t;
  };
  if (is_begin_with(target, "x86_64")) {
    return AddX86ConstantWriterPass(os);
  }
  if (is_begin_with(target, "aarch64")) {
    return AddARMConstantWriterPass(os);
  }
  if (is_begin_with(target, "riscv")) {
    return AddRISCVConstantWriterPass(os);
  }
  return AddGenericCXXConstantWriterPass(os);
}

Pass* PassManager::AddConvertTFCFGPass() { return AddPass<ConvertTFCFG>(); }

Pass* PassManager::AddDCEPass() { return AddPass<DCE>(); }

Pass* PassManager::AddDevicePlacementPass() {
  return AddPass<DevicePlacement>();
}

Pass* PassManager::AddFusionPass(const FusionOptions& opts) {
  return AddPass<Fusion>(opts);
}

Pass* PassManager::AddGenericConstantWriterPass(std::ostream& os,
                                                bool bitcode_format) {
  return AddPass<GenericConstantWriter>(os, bitcode_format);
}

Pass* PassManager::AddGenericCXXConstantWriterPass(std::ostream& os) {
  return AddPass<GenericCXXConstantWriter>(os);
}

Pass* PassManager::AddGenericCXXCodeGenPass(std::ostringstream& os,
                                            std::ostringstream& header_os) {
  return AddPass<GenericCXXCodeGen>(os, header_os);
}

Pass* PassManager::AddGenericCXXCodeGenPass(std::ostringstream& os,
                                            std::ostringstream& header_os,
                                            std::ostream& dynamic_check_os,
                                            const CXXCodeGenOpts& opts) {
  return AddPass<GenericCXXCodeGen>(os, header_os, dynamic_check_os, opts);
}

Pass* PassManager::AddGenericLLVMIRCodeGenPass(
    const std::string& name, ConstantDataStorage constant_data_storage) {
  return AddPass<GenericLLVMIRCodeGen>(name, constant_data_storage);
}

Pass* PassManager::AddGenericLLVMIRCodeGenPass() {
  return AddPass<GenericLLVMIRCodeGen>();
}

Pass* PassManager::AddGenericLLVMIRCodeGenPass(
    ConstantDataStorage constant_data_storage) {
  return AddPass<GenericLLVMIRCodeGen>(constant_data_storage);
}

Pass* PassManager::AddGenericLLVMIRWriterPass(std::ostream& os,
                                              bool bitcode_format) {
  return AddPass<GenericLLVMIRWriter>(os, bitcode_format);
}

Pass* PassManager::AddInputRewriterPass(
    const std::vector<std::string>& inputs) {
  return AddPass<InputRewriter>(inputs);
}

Pass* PassManager::AddInputLegalizerPass(
    int batch_size, const std::vector<std::string>& inputs_shapes,
    const std::string& scale_str) {
  return AddPass<InputLegalizer>(batch_size, inputs_shapes, scale_str);
}

Pass* PassManager::AddInstSimplifyPass() { return AddPass<InstSimplify>(); }

Pass* PassManager::AddInstSimplifyPass(const CXXCodeGenOpts& opts) {
  return AddPass<InstSimplify>(opts);
}

Pass* PassManager::AddLinkPass(const std::ostringstream& obj_code,
                               const std::ostringstream& obj_constants,
                               const std::string& output_file_name,
                               const CXXCodeGenOpts& opts) {
  return AddPass<GenericJITLinker>(obj_code, obj_constants, output_file_name,
                                   opts);
}

Pass* PassManager::AddObjEmitPass(
    std::ostringstream& out, const std::ostringstream& source,
    const std::vector<std::string>& header_searchs,
    const CXXCodeGenOpts& opts) {
  return AddPass<CXXJITCompiler>(out, source, header_searchs, opts);
}

Pass* PassManager::AddONNXExtensionLegalizerPass() {
  return AddPass<ONNXExtensionLegalizer>();
}

Pass* PassManager::AddSerializerPass(std::ostringstream* os,
                                     bool emit_weights) {
  return AddPass<Serializer>(os, emit_weights);
}

Pass* PassManager::AddOutputRewriterPass(
    const std::vector<std::string>& outputs) {
  return AddPass<OutputRewriter>(outputs);
}

Pass* PassManager::AddReorderChannelPass(bool channel_first) {
  return AddPass<ReorderChannel>(channel_first);
}
Pass* PassManager::AddRISCVBinaryWriterPass(std::ostream& os) {
  return AddPass<RISCVBinaryWriter>(os);
}

Pass* PassManager::AddRISCVConstantWriterPass(std::ostream& os) {
  return AddPass<RISCVConstantWriter>(os);
}
Pass* PassManager::AddRISCVLLVMIRCodeGenPass(
    ConstantDataStorage constant_data_storage, const std::string& rt_lib_name) {
  return AddPass<RISCVLLVMIRCodeGen>(constant_data_storage, rt_lib_name);
}

Pass* PassManager::AddRISCVLLVMIRCodeGenPass(
    ConstantDataStorage constant_data_storage) {
  return AddPass<RISCVLLVMIRCodeGen>(constant_data_storage);
}

Pass* PassManager::AddSplittingPass() { return AddPass<Splitting>(); }

Pass* PassManager::AddTemplatedCXXCodeGenPass(std::ostringstream& os,
                                              std::ostringstream& header_os,
                                              const CXXCodeGenOpts& opts) {
  return AddPass<TemplatedCXXCodeGen>(os, header_os, opts);
}

Pass* PassManager::AddTFExtensionLegalizerPass(bool convert_split_to_slice) {
  return AddPass<TFExtensionLegalizer>(convert_split_to_slice);
}

Pass* PassManager::AddTFLiteExtensionLegalizerPass() {
  return AddPass<TFLITEExtensionLegalizer>();
}

Pass* PassManager::AddTritonConfigWriterPass(const std::string& filename,
                                             int max_batch_size) {
  return AddPass<TritonConfigWriter>(filename, max_batch_size);
}

Pass* PassManager::AddTypeCastPass() { return AddPass<TypeCast>(); }

Pass* PassManager::AddTypeLegalizerPass() { return AddPass<TypeLegalizer>(); }

Pass* PassManager::AddTypeLegalizerPass(bool relaxed) {
  return AddPass<TypeLegalizer>(relaxed);
}

Pass* PassManager::AddWeightsQuantizerPass(Quantization quant,
                                           const std::string& file,
                                           const CXXCodeGenOpts& opts) {
  return AddPass<WeightsQuantizer>(quant, file, opts);
}

Pass* PassManager::AddX86BinaryWriterPass(std::ostream& os) {
  return AddPass<X86BinaryWriter>(os);
}

Pass* PassManager::AddX86ConstantWriterPass(std::ostream& os) {
  return AddPass<X86ConstantWriter>(os);
}

Pass* PassManager::AddX86LLVMIRCodeGenPass() {
  return AddPass<X86LLVMIRCodeGen>();
}

Pass* PassManager::AddX86LLVMIRCodeGenPass(
    ConstantDataStorage constant_data_storage) {
  return AddPass<X86LLVMIRCodeGen>(constant_data_storage);
}

Pass* PassManager::AddConstantDecombinePass() {
  return AddPass<ConstantDecombine>();
}

} // end namespace halo
