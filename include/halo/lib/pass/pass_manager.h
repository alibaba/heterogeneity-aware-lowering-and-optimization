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

#include "halo/halo.h"
#include "halo/lib/framework/global_context.h"
#include "halo/lib/ir/module.h"
#include "halo/lib/pass/pass.h"
#include "halo/lib/target/generic_cxx/code_formatter.h"
#include "halo/lib/target/generic_cxx/generic_cxx_codegen.h"
#include "halo/lib/target/generic_llvmir/generic_llvmir_codegen.h"
#include "halo/lib/transforms/fusion.h"

namespace halo {

class FunctionPassManager;
class PassManagerImpl;
struct FusionOptions;

// A class that manages all IR transformation passes.
class HL_API_EXPORT PassManager final {
 public:
  explicit PassManager(GlobalContext& ctx);
  ~PassManager();

  /// Add a pass to the pass manager.
  template <typename T, typename... TS>
  T* AddPass(TS&&... args) {
    auto pass = std::make_unique<T>(std::forward<TS>(args)...);
    T* ret = static_cast<T*>(pass.get());
    Add(std::move(pass));
    return ret;
  }

  /// Apply all passes on `module`.
  Status Run(Module* module);

  /// Print the pass structure.
  void Print(std::ostream& os) const;

  void Dump() const;
  Pass* AddAnalyzerPass(std::ostream* os, const AnalyzerOpts& opts);
  Pass* AddARMBinaryWriterPass(std::ostream& os);
  Pass* AddARMConstantWriterPass(std::ostream& os);
  Pass* AddARMLLVMIRCodeGenPass(ConstantDataStorage constant_data_storage);
  Pass* AddCAFFEExtensionLegalizerPass();
  Pass* AddCodeFormatterPass(std::ostringstream& code,
                             std::ostringstream& header,
                             const CXXCodeGenOpts& opts);
  Pass* AddConstantDecombinePass();
  Pass* AddConstantWriterPass(std::ostream& os, const std::string& target);
  Pass* AddConvertTFCFGPass();
  Pass* AddDCEPass();
  Pass* AddDevicePlacementPass();
  Pass* AddFusionPass(const FusionOptions& opts);
  Pass* AddGenericConstantWriterPass(std::ostream& os, bool bitcode_format);
  Pass* AddGenericCXXConstantWriterPass(std::ostream& os);
  Pass* AddGenericCXXCodeGenPass(std::ostringstream& os,
                                 std::ostringstream& header_os);
  Pass* AddGenericCXXCodeGenPass(std::ostringstream& os,
                                 std::ostringstream& header_os,
                                 std::ostream& dynamic_check_os,
                                 const CXXCodeGenOpts& opts);
  Pass* AddGenericLLVMIRCodeGenPass();
  Pass* AddGenericLLVMIRCodeGenPass(ConstantDataStorage constant_data_storage);
  Pass* AddGenericLLVMIRCodeGenPass(const std::string& name,
                                    ConstantDataStorage constant_data_storage);
  Pass* AddGenericLLVMIRWriterPass(std::ostream& os, bool bitcode_format);
  Pass* AddInputLegalizerPass(int batch_size,
                              const std::vector<std::string>& inputs_shapes,
                              const std::string& scale_str);
  Pass* AddInputRewriterPass(const std::vector<std::string>& inputs);
  Pass* AddInstSimplifyPass();
  Pass* AddInstSimplifyPass(const CXXCodeGenOpts& opts);
  Pass* AddLinkPass(const std::ostringstream& obj_code,
                    const std::ostringstream& obj_constants,
                    const std::string& output_file_name,
                    const CXXCodeGenOpts& opts);
  Pass* AddObjEmitPass(std::ostringstream& out,
                       const std::ostringstream& source,
                       const std::vector<std::string>& header_searchs,
                       const CXXCodeGenOpts& opts);
  Pass* AddONNXExtensionLegalizerPass();
  Pass* AddOutputRewriterPass(const std::vector<std::string>& outputs);
  Pass* AddReorderChannelPass(bool channel_first);
  Pass* AddRISCVBinaryWriterPass(std::ostream& os);
  Pass* AddRISCVConstantWriterPass(std::ostream& os);
  Pass* AddRISCVLLVMIRCodeGenPass(ConstantDataStorage constant_data_storage);

  Pass* AddRISCVLLVMIRCodeGenPass(ConstantDataStorage constant_data_storage,
                                  const std::string& rt_lib_name);
  Pass* AddSerializerPass(std::ostringstream* os, bool emit_weights);
  Pass* AddSplittingPass();
  Pass* AddTemplatedCXXCodeGenPass(std::ostringstream& os,
                                   std::ostringstream& header_os,
                                   const CXXCodeGenOpts& opts);
  Pass* AddTFExtensionLegalizerPass(bool convert_split_to_slice);
  Pass* AddTFLiteExtensionLegalizerPass();
  Pass* AddTritonConfigWriterPass(const std::string& filename,
                                  int max_batch_size);
  Pass* AddTypeCastPass();
  Pass* AddTypeLegalizerPass();
  Pass* AddTypeLegalizerPass(bool relaxed);
  Pass* AddWeightsQuantizerPass(Quantization quant, const std::string& file,
                                const CXXCodeGenOpts& opts);
  Pass* AddX86BinaryWriterPass(std::ostream& os);
  Pass* AddX86ConstantWriterPass(std::ostream& os);
  Pass* AddX86LLVMIRCodeGenPass();
  Pass* AddX86LLVMIRCodeGenPass(ConstantDataStorage constant_data_storage);
  GlobalContext& GetGlobalContext() const;

 private:
  Pass* Add(std::unique_ptr<ModulePass> pass);
  Pass* Add(std::unique_ptr<FunctionPass> pass);
  Pass* Add(std::unique_ptr<BasicBlockPass> pass);
  std::unique_ptr<PassManagerImpl> impl_;
};

} // end namespace halo.

#endif // HALO_LIB_PASS_PASS_MGR_H_
