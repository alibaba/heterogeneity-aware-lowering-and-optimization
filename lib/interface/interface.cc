//===- interface.cc -------------------------------------------------------===//
//
// Copyright (C) 2019-2021 Alibaba Group Holding Limited.
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
#include <fstream>
#include <string>
#include <vector>

#include "halo/halo.h"
#include "halo/lib/framework/common.h"
#include "halo/lib/framework/global_context.h"
#include "halo/lib/ir/ir_builder.h"
#include "halo/lib/parser/parser.h"
#include "halo/lib/pass/pass_manager.h"
#include "halo/lib/transforms/fusion.h"
#include "halo/utils/passes_helper.h"

namespace halo {
HL_API_EXPORT
int Compile(ModelFormat format, const std::vector<const char*>& models,
            const std::vector<size_t>& model_sizes, const std::string& name,
            const std::string& temp_dir, const std::string& target, int batch,
            const std::vector<std::string>& input_shapes,
            const std::vector<std::string>& inputs,
            const std::vector<std::string>& outputs,
            const CXXCodeGenOpts& cg_opts) {
  GlobalContext ctx;
  ctx.SetTargetTriple(std::string(target));
  if (target.substr(0, 3) == "cxx") {
    ctx.SetTargetTriple("x86_64"); // For binary constant writer.
  }

  Module m(ctx, "halo_module");

  FunctionBuilder func_builder(&m);
  std::string func_name = "model";
  Function* func = func_builder.CreateFunction(func_name);
  if (auto status = Parser::Parse(func, models, model_sizes, format);
      status != Status::SUCCESS) {
    return 1;
  }
  PassManager pm(ctx);
  FusionOptions fusion_opts;
  PopulateOptPasses(&pm, target, input_shapes, inputs, outputs, batch, "",
                    ChannelOrder::None, false, false, ModelFormat::TENSORFLOW,
                    cg_opts, fusion_opts);
  std::string code_fn = temp_dir + "/" + name + ".cc";
  std::string weights_fn = temp_dir + "/" + name + ".bin";
  std::string header_fn = temp_dir + "/" + name + ".h";

  std::ofstream out_code(code_fn, std::ofstream::binary);
  std::ofstream out_constants(weights_fn, std::ofstream::binary);
  std::ofstream out_header(header_fn, std::ofstream::binary);
  bool is_c_or_cxx_output =
      target.substr(0, 3) == "cxx" || target.substr(0, 2) == "cc";

  PopulateCodeGenPasses(&pm, &out_code, &out_constants, &out_header, &std::cerr,
                        target, is_c_or_cxx_output, false /*is_binary_output*/,
                        false /* EmitDataAsC */, false /*EmitCodeOnly*/,
                        false /* EmitLLVMIR */, false /* EmitTritonConfig */,
                        "" /*TritonConfigFile */, Quantization::None,
                        "" /* PGQFile */, false /* RISCVOpt */, cg_opts);
  pm.Run(&m);
  return 0;
}

HL_API_EXPORT
int CompileTFGraph(const char* graphdef, size_t graphdef_size,
                   const std::string& name, const std::string& temp_dir,
                   const std::vector<std::string>& input_shapes,
                   const CXXCodeGenOpts& cg_opts) {
  return Compile(ModelFormat::TENSORFLOW, {graphdef}, {graphdef_size}, name,
                 temp_dir, "cxx", 0, input_shapes, {}, {}, cg_opts);
}

} // namespace halo
