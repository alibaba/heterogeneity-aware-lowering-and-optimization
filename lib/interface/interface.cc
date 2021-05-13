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

static std::tuple<std::unique_ptr<Module>, Function*> CreateModule(
    GlobalContext* ctx, const std::string& target) {
  ctx->SetTargetTriple(std::string(target));
  if (target.substr(0, 3) == "cxx") {
    ctx->SetTargetTriple("x86_64"); // For binary constant writer.
  }

  std::unique_ptr<Module> m = std::make_unique<Module>(*ctx, "halo_module");

  FunctionBuilder func_builder(m.get());
  std::string func_name = "model";
  Function* func = func_builder.CreateFunction(func_name);
  return std::make_tuple(std::move(m), func);
}

static int InvokeCompiler(Module* m, const std::string& name,
                          const std::string& temp_dir,
                          const std::string& target, int batch,
                          const std::vector<std::string>& input_shapes,
                          const std::vector<std::string>& inputs,
                          const std::vector<std::string>& outputs,
                          const CXXCodeGenOpts& cg_opts) {
  PassManager pm(m->GetGlobalContext());
  FusionOptions fusion_opts;
  PopulateOptPasses(&pm, target, input_shapes, inputs, outputs, batch, "",
                    ChannelOrder::None, false, false, ModelFormat::TENSORFLOW,
                    cg_opts, fusion_opts);
  std::string ext = (cg_opts.dialect == halo::Dialect::C99) ? ".c" : ".cc";
  std::string code_fn = temp_dir + "/" + name + ext;
  std::string weights_fn = temp_dir + "/" + name + ".bin";

  std::string header_fn = temp_dir + "/" + name + ".h";

  std::ofstream out_code(code_fn, std::ofstream::binary);
  std::ofstream out_constants(weights_fn, std::ofstream::binary);
  std::ofstream out_header(header_fn, std::ofstream::binary);
  bool is_c_or_cxx_output =
      target.substr(0, 3) == "cxx" || target.substr(0, 2) == "cc";
  std::ostringstream buf_code;
  std::ostringstream buf_header;
  PopulateCodeGenPasses(&pm, &buf_code, &out_constants, &buf_header, &std::cerr,
                        target, is_c_or_cxx_output, false /*is_binary_output*/,
                        false /* EmitDataAsC */, false /*EmitCodeOnly*/,
                        false /* EmitLLVMIR */, false /* EmitTritonConfig */,
                        "" /*TritonConfigFile */, Quantization::None,
                        "" /* PGQFile */, false /* RISCVOpt */, cg_opts);
  pm.Run(m);
  out_code << buf_code.str();
  out_header << buf_header.str();
  return 0;
}

HL_API_EXPORT
int Compile(ModelFormat format, const std::vector<const void*>& model_defs,
            const std::string& name, const std::string& temp_dir,
            const std::string& target, int batch,
            const std::vector<std::string>& input_shapes,
            const std::vector<std::string>& inputs,
            const std::vector<std::string>& outputs,
            const CXXCodeGenOpts& cg_opts) {
  GlobalContext ctx;
  Function* func;
  std::unique_ptr<Module> m;
  std::tie(m, func) = CreateModule(&ctx, target);

  if (auto status = Parser::Parse(func, model_defs, format);
      status != Status::SUCCESS) {
    return 1;
  }

  return InvokeCompiler(m.get(), name, temp_dir, target, batch, input_shapes,
                        inputs, outputs, cg_opts);
}

HL_API_EXPORT
int Compile(ModelFormat format, const std::vector<const char*>& models,
            const std::vector<size_t>& model_sizes, const std::string& name,
            const std::string& temp_dir, const std::string& target, int batch,
            const std::vector<std::string>& input_shapes,
            const std::vector<std::string>& inputs,
            const std::vector<std::string>& outputs,
            const CXXCodeGenOpts& cg_opts) {
  GlobalContext ctx;
  Function* func;
  std::unique_ptr<Module> m;
  std::tie(m, func) = CreateModule(&ctx, target);
  if (auto status = Parser::Parse(func, models, model_sizes, format);
      status != Status::SUCCESS) {
    return 1;
  }

  return InvokeCompiler(m.get(), name, temp_dir, target, batch, input_shapes,
                        inputs, outputs, cg_opts);
}

HL_API_EXPORT
int CompileTFGraph(const void* graphdef, const std::string& name,
                   const std::string& temp_dir,
                   const std::vector<std::string>& input_shapes,
                   const CXXCodeGenOpts& cg_opts) {
  return Compile(ModelFormat::TENSORFLOW, {graphdef}, name, temp_dir, "cxx", 0,
                 input_shapes, {}, {}, cg_opts);
}

HL_API_EXPORT
int CompileTFGraph(const char* pb_buf, size_t pb_buf_size,
                   const std::string& name, const std::string& temp_dir,
                   const std::vector<std::string>& input_shapes,
                   const CXXCodeGenOpts& cg_opts) {
  return Compile(ModelFormat::TENSORFLOW, {pb_buf}, {pb_buf_size}, name,
                 temp_dir, "cxx", 0, input_shapes, {}, {}, cg_opts);
}

} // namespace halo
