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
#include "halo/lib/transforms/analyzer.h"
#include "halo/lib/transforms/fusion.h"
#include "halo/utils/passes_helper.h"
#include "halo/utils/path.h"

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

static int InvokeCompiler(Module* m, const std::string& target, int batch,
                          const std::vector<std::string>& input_shapes,
                          const std::vector<std::string>& inputs,
                          const std::vector<std::string>& outputs,
                          ModelFormat model_format,
                          const CXXCodeGenOpts& cg_opts,
                          const std::string& main_output_file_name,
                          ModelInfo* model_info, bool is_compile_model = true,
                          const int model_type = 0) {
  auto& ctx = m->GetGlobalContext();
  ctx.SetVerbosity(1);
  ctx.SetBasePath(GetBaseDir());
  ctx.SetODLAIncludePath(FindODLAIncPath(ctx.GetBasePath(), {}));
  ctx.SetODLALibraryPath(
      FindODLALibPath(ctx.GetBasePath(), {}, cg_opts.linked_odla_lib));

  PassManager pm(ctx);
  FusionOptions fusion_opts;

  if (is_compile_model) {
    PopulateOptPasses(&pm, target, input_shapes, inputs, outputs, batch, "",
                      false, model_format, cg_opts, fusion_opts);
    std::string ext = (cg_opts.dialect == halo::Dialect::C99) ? ".c" : ".cc";
    bool is_c_or_cxx_output =
        target.substr(0, 3) == "cxx" || target.substr(0, 2) == "cc";
    std::ostringstream buf_code;
    std::ostringstream buf_constants;
    std::ostringstream buf_header;

    PopulateCodeGenPasses(
        &pm, &buf_code, &buf_constants, &buf_header, &std::cerr, target,
        is_c_or_cxx_output, false /*is_binary_output*/, false /* EmitDataAsC */,
        false /*EmitCodeOnly*/, false /* EmitLLVMIR */,
        false /* EmitTritonConfig */, "" /*TritonConfigFile */,
        Quantization::None, "" /* PGQFile */, false /* RISCVOpt */, cg_opts,
        main_output_file_name);
    pm.Run(m);

    if (!cg_opts.emit_shared_lib) {
      std::ofstream out_code(main_output_file_name, std::ofstream::binary);
      out_code << buf_code.str();
    }
    if (!cg_opts.emit_shared_lib) {
      auto weights_fn = GetDerivedFileName(main_output_file_name, ".bin");
      std::ofstream out_constants(weights_fn, std::ofstream::binary);
      out_constants << buf_constants.str();
    }
    if (cg_opts.emit_header) {
      auto header_fn = GetDerivedFileName(main_output_file_name, ".h");
      std::ofstream out_header(header_fn, std::ofstream::binary);
      out_header << buf_header.str();
    }
    if (model_info != nullptr) {
      *model_info = ctx.GetModelInfo();
    }
  } else { // model analysis
    PopulateOptPasses(&pm, target, input_shapes, inputs, outputs, 1, "", false,
                      model_format, cg_opts, fusion_opts);
    AnalyzerOpts alz_opts;
    alz_opts.batch_size = model_info->adaptive_bsz;
    alz_opts.print_details = false;
    alz_opts.qps = model_info->input_qps;
    alz_opts.model_type = model_type;
    Analyzer* analyzer =
        static_cast<Analyzer*>(pm.AddAnalyzerPass(&std::cout, alz_opts));
    pm.Run(m);
    if (model_info != nullptr) {
      int bsz = 1;
      std::string& s = analyzer->GetReourceEst(bsz);
      size_t sz = s.size();
      if (sz >= HALO_VODLA_MAX_OUTPUT_RSC_EST) {
        return 1;
      } else {
        s.copy(model_info->output_rsc_est, sz);
        model_info->output_rsc_est[sz] = '\0';
        model_info->adaptive_bsz = bsz;
      }
    }
  }

  return 0;
}

HL_API_EXPORT
int Compile(ModelFormat format, const std::vector<const void*>& model_defs,
            const std::string& target, int batch,
            const std::vector<std::string>& input_shapes,
            const std::vector<std::string>& inputs,
            const std::vector<std::string>& outputs,
            const CXXCodeGenOpts& cg_opts,
            const std::string& main_output_file_name, ModelInfo* model_info) {
  GlobalContext ctx;
  Function* func;
  std::unique_ptr<Module> m;
  std::tie(m, func) = CreateModule(&ctx, target);

  if (auto status = Parser::Parse(func, model_defs, format);
      status != Status::SUCCESS) {
    return 1;
  }

  return InvokeCompiler(m.get(), target, batch, input_shapes, inputs, outputs,
                        format, cg_opts, main_output_file_name, model_info);
}

HL_API_EXPORT
int Compile(halo::ModelFormat format, const std::vector<const char*>& models,
            const std::vector<size_t>& model_sizes, const std::string& target,
            int batch, const std::vector<std::string>& input_shapes,
            const std::vector<std::string>& inputs,
            const std::vector<std::string>& outputs,
            const CXXCodeGenOpts& cg_opts,
            const std::string& main_output_file_name, ModelInfo* model_info,
            bool is_compile_model, const int model_type) {
  GlobalContext ctx;
  Function* func;
  std::unique_ptr<Module> m;
  std::tie(m, func) = CreateModule(&ctx, target);
  if (auto status = Parser::Parse(func, models, model_sizes, format);
      status != Status::SUCCESS) {
    return 1;
  }

  return InvokeCompiler(m.get(), target, batch, input_shapes, inputs, outputs,
                        format, cg_opts, main_output_file_name, model_info,
                        is_compile_model, model_type);
}

HL_API_EXPORT
int Compile(halo::ModelFormat format, const std::vector<const char*>& models,
            const std::vector<size_t>& model_sizes, const std::string& target,
            int batch, const std::vector<std::string>& input_shapes,
            const std::vector<std::string>& inputs,
            const std::vector<std::string>& outputs,
            const CXXCodeGenOpts& cg_opts,
            const std::string& main_output_file_name, ModelInfo* model_info,
            bool is_compile_model) {
  GlobalContext ctx;
  Function* func;
  std::unique_ptr<Module> m;
  std::tie(m, func) = CreateModule(&ctx, target);
  if (auto status = Parser::Parse(func, models, model_sizes, format);
      status != Status::SUCCESS) {
    return 1;
  }

  return InvokeCompiler(m.get(), target, batch, input_shapes, inputs, outputs,
                        format, cg_opts, main_output_file_name, model_info,
                        is_compile_model);
}

HL_API_EXPORT
int CompileTFGraph(const void* graphdef,
                   const std::vector<std::string>& input_shapes, int batch,
                   const CXXCodeGenOpts& cg_opts,
                   const std::string& main_output_file, ModelInfo* model_info) {
  return Compile(ModelFormat::TENSORFLOW, {graphdef}, "cxx", batch,
                 input_shapes, {}, {}, cg_opts, main_output_file, model_info);
}

HL_API_EXPORT
int CompileTFGraph(const char* pb_buf, size_t pb_buf_size,
                   const std::vector<std::string>& input_shapes, int batch,
                   const CXXCodeGenOpts& cg_opts,
                   const std::string& main_output_file, ModelInfo* model_info,
                   bool is_compile_model = true) {
  const char* str = main_output_file.c_str();
  const std::string new_str(str);
  return Compile(ModelFormat::TENSORFLOW, {pb_buf}, {pb_buf_size}, "cxx", batch,
                 input_shapes, {}, {}, cg_opts, new_str, model_info,
                 is_compile_model);
}

} // namespace halo

// NOLINTNEXTLINE
static std::vector<std::string> ToStrings(size_t n, const char* const strs[]) {
  std::vector<std::string> strs_v;
  strs_v.reserve(n);
  for (unsigned i = 0; i < n; ++i) {
    strs_v.push_back(std::string(strs[i])); // NOLINT
  }
  return strs_v;
}

HL_API_EXPORT
int halo_Compile(halo::ModelFormat model_format, unsigned num_models,
                 const char* const models[], size_t const model_sizes[],
                 const char* target, int batch, unsigned num_input_shapes,
                 const char* const input_shapes[], unsigned num_inputs,
                 const char* const inputs[], unsigned num_outputs,
                 const char* const outputs[], const HaloCodeGenOpts* cg_opts,
                 const char* main_output_file, HaloModelInfo* model_info) {
  const halo::CXXCodeGenOpts& opts =
      *reinterpret_cast<const halo::CXXCodeGenOpts*>(cg_opts);
  std::vector<const char*> models_data(num_models);
  std::vector<size_t> models_sizes(num_models);
  for (unsigned i = 0; i < num_models; ++i) {
    models_data[i] = models[i];
    models_sizes[i] = model_sizes[i];
  }
  return halo::Compile(
      model_format, models_data, models_sizes, std::string(target), batch,
      ToStrings(num_input_shapes, input_shapes), ToStrings(num_inputs, inputs),
      ToStrings(num_outputs, outputs), opts, std::string(main_output_file),
      model_info, true);
}

HL_API_EXPORT
// NOLINTNEXTLINE
int halo_CompileTFPbGraph(const char* pb_buf, size_t pb_buf_size,
                          size_t num_input_shapes, const char* input_shapes[],
                          int batch, const HaloCodeGenOpts* cg_opts,
                          const char* main_output_file,
                          HaloModelInfo* model_info) {
  const halo::CXXCodeGenOpts& opts =
      *reinterpret_cast<const halo::CXXCodeGenOpts*>(cg_opts);
  return halo::CompileTFGraph(pb_buf, pb_buf_size,
                              ToStrings(num_input_shapes, input_shapes), batch,
                              opts, std::string(main_output_file), model_info);
}

HL_API_EXPORT
int halo_Analyze(halo::ModelFormat model_format, unsigned num_models,
                 const char* const models[], size_t const model_sizes[],
                 const char* target, int batch, unsigned num_input_shapes,
                 const char* const input_shapes[], unsigned num_inputs,
                 const char* const inputs[], unsigned num_outputs,
                 const char* const outputs[], const HaloCodeGenOpts* cg_opts,
                 const char* main_output_file, HaloModelInfo* model_info,
                 const int model_type) {
  const halo::CXXCodeGenOpts& opts =
      *reinterpret_cast<const halo::CXXCodeGenOpts*>(cg_opts);
  std::vector<const char*> models_data(num_models);
  std::vector<size_t> models_sizes(num_models);
  for (unsigned i = 0; i < num_models; ++i) {
    models_data[i] = models[i];
    models_sizes[i] = model_sizes[i];
  }
  return halo::Compile(
      model_format, models_data, models_sizes, std::string(target), batch,
      ToStrings(num_input_shapes, input_shapes), ToStrings(num_inputs, inputs),
      ToStrings(num_outputs, outputs), opts, std::string(main_output_file),
      model_info, false, model_type);
}