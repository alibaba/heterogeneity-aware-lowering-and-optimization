//===- driver.cc ----------------------------------------------------------===//
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

#include <fstream>
#include <set>
#include <string>

#include "halo/lib/framework/common.h"
#include "halo/lib/ir/ir_builder.h"
#include "halo/lib/parser/parser.h"
#include "halo/lib/pass/pass_manager.h"
#include "halo/lib/transforms/fusion.h"
#include "halo/lib/transforms/reorder_channel.h"
#include "halo/utils/cl_options.h"
#include "halo/utils/passes_helper.h"
#include "halo/version.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Program.h"

using namespace halo;

static llvm::cl::opt<std::string> Target(
    "target", llvm::cl::desc("target triple"),
    llvm::cl::init("x86_64-unknown-linux"));
static llvm::cl::opt<std::string> Processor("processor",
                                            llvm::cl::desc("processor name"),
                                            llvm::cl::init("native"));
static llvm::cl::opt<std::string> OutputFile(
    "o", llvm::cl::desc("output file name."), llvm::cl::Required);

static llvm::cl::opt<bool> PrintAll(
    "print-all", llvm::cl::desc("print intermediates of all passes"),
    llvm::cl::init(false));

static llvm::cl::opt<bool> PrintPass("print-pass",
                                     llvm::cl::desc("print pass name"),
                                     llvm::cl::init(false));

static llvm::cl::opt<bool> EmitLLVMIR("emit-llvm",
                                      llvm::cl::desc("output the LLVM IR code"),
                                      llvm::cl::init(false));
static llvm::cl::opt<std::string> ModuleName("module-name",
                                             llvm::cl::desc("name of module"),
                                             llvm::cl::init("halo_module"));

static llvm::cl::opt<ReorderChannel::ChannelOrder> ReorderChannelLayout(
    llvm::cl::values(clEnumValN(ReorderChannel::ChannelOrder::None, "none",
                                "No reordering"),
                     clEnumValN(ReorderChannel::ChannelOrder::ChannelFirst,
                                "channel-first", "Reorder to channel first"),
                     clEnumValN(ReorderChannel::ChannelOrder::ChannelLast,
                                "channel-last", "Reorder to channel last")),
    "reorder-data-layout", llvm::cl::desc("Reorder the data layout"),
    llvm::cl::init(ReorderChannel::ChannelOrder::None));

static llvm::cl::opt<bool> RemoveInputTranspose(
    "remove-input-transpose", llvm::cl::desc("Remove the transpose for inputs"),
    llvm::cl::init(false));

static llvm::cl::opt<bool> RemoveOutputTranspose(
    "remove-output-transpose",
    llvm::cl::desc("Remove the transpose for outputs"), llvm::cl::init(false));

static llvm::cl::opt<bool> SeparateConstants(
    "separate-constants",
    llvm::cl::desc("Generate separate file for constants"),
    llvm::cl::init(true));
static llvm::cl::opt<bool> DisableBroadcasting(
    "disable-broadcasting", llvm::cl::desc("disable broadcasting of constants"),
    llvm::cl::init(false));
static llvm::cl::opt<bool> DisableConvBN(
    "disable-convert-bn",
    llvm::cl::desc("disable convert Batch Normalization into mul/add"),
    llvm::cl::init(false));
static llvm::cl::opt<bool> EmitCodeOnly(
    "code-only", llvm::cl::desc("Generate the code only"),
    llvm::cl::init(false));

static llvm::cl::opt<bool> RISCVOpt(
    "riscv-opt", llvm::cl::desc("Enable optimizations for RISC-V only"),
    llvm::cl::init(false));

static llvm::cl::opt<CodeGen::BF16Mode> BF16Mode(
    llvm::cl::values(
        clEnumValN(CodeGen::BF16Mode::Disable, "disable", "disable bf16 mode"),
        clEnumValN(CodeGen::BF16Mode::Accuracy, "accuracy", "white list Model"),
        clEnumValN(CodeGen::BF16Mode::Performace, "performace",
                   "global enable bf16,except black list"),
        clEnumValN(CodeGen::BF16Mode::Auto, "auto", "automixprecision")),
    "bf16-mode", llvm::cl::desc("Enable BF16 with acc/perf/auto mode"),
    llvm::cl::init(CodeGen::BF16Mode::Disable));

static llvm::cl::opt<bool> EnableFP16("enable-fp16",
                                      llvm::cl::desc("Enable FP16 mode"),
                                      llvm::cl::init(false));

static llvm::cl::opt<bool> EnableIpuDevice("enable-ipu-device",
                                           llvm::cl::desc("Enable IPU Device"),
                                           llvm::cl::init(false));

static llvm::cl::opt<bool> UseIpuModel("use-ipu-model",
                                       llvm::cl::desc("Use IPU Mode"),
                                       llvm::cl::init(false));

static llvm::cl::opt<int> IpuNum(
    "ipu-num",
    llvm::cl::desc("Num of ipus, should consistent with subgraph num"),
    llvm::cl::init(1));

static llvm::cl::opt<int> BatchesPerStep(
    "batches-per-step", llvm::cl::desc("Specify batches num for each step"),
    llvm::cl::init(1));

static llvm::cl::opt<bool> DisableCodeFormat(
    "disable-code-format",
    llvm::cl::desc("Disable formatting the generated C/C++ code"),
    llvm::cl::init(false));

static llvm::cl::opt<CodeGen::ExecMode> ExecMode(
    llvm::cl::values(clEnumValN(CodeGen::ExecMode::Compile, "compile",
                                "Compilation-Execution Model"),
                     clEnumValN(CodeGen::ExecMode::Interpret, "interpret",
                                "Interpreter Model")),
    "exec-mode", llvm::cl::desc("Execution model of emitted code"),
    llvm::cl::init(CodeGen::ExecMode::Compile));

static llvm::cl::opt<bool> EmitDataAsC(
    "emit-data-as-c", llvm::cl::desc("Emit Constants as C/C++ code"),
    llvm::cl::init(false));

static llvm::cl::opt<bool> PrintMemStats(
    "print-mem-stats", llvm::cl::desc("Print Memory Usage Stats"),
    llvm::cl::init(false));

static llvm::cl::opt<bool> EmitValueReset(
    "emit-value-reset",
    llvm::cl::desc("Emit code to reset value life cycle ends"),
    llvm::cl::init(false));

static llvm::cl::opt<bool> EmitValueIDAsInt(
    "emit-value-id-as-int",
    llvm::cl::desc("Emit value id as integer. (default is string"),
    llvm::cl::init(false));

static llvm::cl::opt<bool> SplitFunction(
    "fiss-function",
    llvm::cl::desc("Split the function into multiple subfunctions"),
    llvm::cl::init(false));

static llvm::cl::opt<CodeGen::API> Api(
    llvm::cl::values(clEnumValN(CodeGen::API::HALO_RT, "halo_rt",
                                "Using Halo Runtime Library"),
                     clEnumValN(CodeGen::API::ODLA_05, "odla_05",
                                "Using ODLA 0.5")),
    "api", llvm::cl::desc("APIs used in emitted code"),
    llvm::cl::init(CodeGen::API::ODLA_05));

static llvm::cl::opt<bool> EmitInferenceFunctionSignature(
    "emit-inference-func-sig",
    llvm::cl::desc("Emit fuction with a universal signature in c/c++ codegen"),
    llvm::cl::init(false));

static llvm::cl::opt<bool> EmitTritonConfig(
    "emit-triton-config",
    llvm::cl::desc("Emit triton inference server config file"),
    llvm::cl::init(false));

static llvm::cl::opt<std::string> TritonConfigFile(
    "triton-config-file", llvm::cl::desc("Triton inference server config file"),
    llvm::cl::init("config.pbtxt"));

static llvm::cl::list<std::string> Inputs(
    "inputs",
    llvm::cl::desc("Specify input names like -inputs=foo -inputs=bar"));

static llvm::cl::list<std::string> Outputs(
    "outputs",
    llvm::cl::desc("Specify output names like -outputs=foo, -outputs=bar:0"));

static llvm::cl::opt<CodeGen::Quantization> QuantWeights(
    llvm::cl::values(clEnumValN(CodeGen::Quantization::QUINT8, "quint8",
                                "Quantize weigths as quint8")),
    "quantize-weights", llvm::cl::desc("Emit weights as quantized"),
    llvm::cl::init(CodeGen::Quantization::None));

static llvm::cl::opt<bool> DisableTypeCast(
    "disable-type-cast", llvm::cl::desc("Disable casting int64 to int32"),
    llvm::cl::init(true));

static llvm::cl::opt<signed> MaxBatch("max-batch-size",
                                      llvm::cl::desc("Specify max batch size"),
                                      llvm::cl::init(8));

static llvm::cl::opt<signed> MinBatch("min-batch-size",
                                      llvm::cl::desc("Specify min batch size"),
                                      llvm::cl::init(1));

static llvm::cl::opt<signed> OptBatch("opt-batch-size",
                                      llvm::cl::desc("Specify opt batch size"),
                                      llvm::cl::init(4));
static llvm::cl::opt<std::string> PGQFile(
    "pgq-file", llvm::cl::init(""),
    llvm::cl::desc("Profiling file for quantization of biases"));

static llvm::cl::opt<bool> CheckModel("check-model",
                                      llvm::cl::desc("dynamic check model"),
                                      llvm::cl::init(false));

#undef HALO_FUSION_OPTIONS
#define HALO_FUSION_CMD_OPTIONS_DECL
#include "halo/lib/ir/fusion.cc.inc"
#undef HALO_FUSION_CMD_OPTIONS_DECL

static bool FormatCode(const std::string& filename) {
  if (filename.empty() || filename == "-") {
    return false;
  }
  // Search clang-format in PATH env.
  auto exe = llvm::sys::findProgramByName("clang-format", {});
  if (!exe) {
    exe = llvm::sys::findProgramByName("clang-format-9", {});
  }
  std::string ret_msg;
  if (exe) {
    ret_msg = "";
    const char* arg0 = "--style=LLVM";
    const char* arg1 = "-i"; // in-place format.
    constexpr int timeout = 10;
    llvm::sys::ExecuteAndWait(exe.get(), {arg0, arg1, filename}, {}, {},
                              timeout, 0, &ret_msg);
  } else {
    ret_msg = "Unable to find formatting tool";
  }
  if (!ret_msg.empty()) {
    std::cerr << "Code format failed: " << ret_msg << "\n";
  }
  return true;
}

static void PrintVersion(llvm::raw_ostream& os) {
  os << "  Version:\t" << HALO_MAJOR << '.' << HALO_MINOR << '.' << HALO_PATCH
     << '\n';
#ifdef HALO_REVISION
  os << "  HALO Repo:" << HALO_REPOSITORY << " Rev:" << HALO_REVISION << '\n';
#endif
#ifdef ODLA_REVISION
  os << "  ODLA Repo:" << ODLA_REPOSITORY << " Rev:" << ODLA_REVISION << '\n';
#endif
#ifndef NDEBUG
  os << "  Build:\tDebug\n";
#else
  os << "  Build:\tRelease\n";
#endif
}

int main(int argc, char** argv) {
  llvm::cl::SetVersionPrinter(PrintVersion);
  llvm::cl::ParseCommandLineOptions(argc, argv);
  GlobalContext ctx;
  ctx.SetBasePath(argv[0]);
  ctx.SetTargetTriple(Target);
  ctx.SetProcessorName(Processor);
  ctx.SetPrintPass(PrintPass);

  Module m(ctx, ModuleName);

  armory::Opts opts;
  Parser::Format format = Parser::Format::INVALID;
  if (ParseModels(ModelFiles, ModelFormat, EntryFunctionName, opts, &m,
                  &format) != Status::SUCCESS) {
    return 1;
  }

  if (PrintAll) {
    m.Dump();
  }

  PassManager pm(ctx);

  std::ofstream of_code;
  std::ofstream of_constants;
  std::ofstream of_header;
  std::ofstream of_dynamic_check;
  std::ostream* out_code = &std::cout;
  std::ostream* out_constants = &std::cout;
  std::ostream* out_header = &std::cout;
  std::ostream* out_dynamic_check = &std::cout;

  bool is_binary_output = false;
  llvm::StringRef target_name(Target);
  bool is_c_or_cxx_output =
      target_name.startswith_lower("cxx") || target_name.startswith_lower("cc");
  llvm::SmallString<128> header_file_name("");
  if (!OutputFile.empty() && OutputFile != "-") {
    of_code.open(OutputFile, std::ofstream::binary);
    out_code = &of_code;
    llvm::StringRef name(OutputFile);
    llvm::SmallString<128> data_file_name(name);
    header_file_name = name;
    is_binary_output = name.endswith(".bc") || name.endswith(".o");
    if (EmitDataAsC) {
      std::string ext =
          llvm::StringRef(Target).startswith_lower("cc") ? "data.c" : "data.cc";
      llvm::sys::path::replace_extension(data_file_name, ext);
    } else {
      llvm::sys::path::replace_extension(data_file_name, ".bin");
    }
    llvm::sys::path::replace_extension(header_file_name, ".h");

    of_constants.open(data_file_name.str(), std::ofstream::binary);
    out_constants = &of_constants;
    of_header.open(header_file_name.str());
    out_header = &of_header;
  }

  if (EmitTritonConfig) {
    if (!TritonConfigFile.empty() &&
        llvm::sys::path::filename(TritonConfigFile).equals(TritonConfigFile)) {
      llvm::SmallString<128> file_name;
      llvm::sys::path::append(file_name,
                              llvm::sys::path::parent_path(OutputFile),
                              TritonConfigFile);
      TritonConfigFile = file_name.str();
    }
  }

  if (CheckModel) {
    llvm::StringRef name(OutputFile);
    llvm::SmallString<128> filename(name);
    llvm::sys::path::replace_extension(filename, ".main.cc.in");
    of_dynamic_check.open(filename.str(), std::ofstream::binary);
    out_dynamic_check = &of_dynamic_check;
  }

  Opts cg_opts;
  cg_opts.bf16_mode = BF16Mode;
  cg_opts.print_mem_stats = PrintMemStats;
  cg_opts.emit_value_reset = EmitValueReset;
  cg_opts.exec_mode = ExecMode.getValue();
  cg_opts.emit_value_id_as_int = EmitValueIDAsInt;
  cg_opts.emit_inference_func_sig = EmitInferenceFunctionSignature;
  cg_opts.emit_dynamic_batch = (Batch.getValue() == kDynamicBatchSize);
  cg_opts.fp16_mode = EnableFP16;
  cg_opts.max_batch_size = MaxBatch.getValue();
  cg_opts.min_batch_size = MinBatch.getValue();
  cg_opts.opt_batch_size = OptBatch.getValue();
  cg_opts.check_model = CheckModel;
  cg_opts.enable_ipu_device = EnableIpuDevice;
  cg_opts.use_ipu_model = UseIpuModel;
  cg_opts.ipu_num = IpuNum;
  cg_opts.batches_per_step = BatchesPerStep;
  cg_opts.api = Api;
  cg_opts.disable_broadcasting = DisableBroadcasting;
  cg_opts.separate_constants = SeparateConstants;
  cg_opts.disable_conv_bn = DisableConvBN;
  cg_opts.remove_input_transpose = RemoveInputTranspose;
  cg_opts.remove_output_transpose = RemoveOutputTranspose;

  if (is_c_or_cxx_output) {
    ctx.SetTargetTriple("x86_64"); // For binary constant writer.
    if (llvm::StringRef(Target).startswith_lower("cc")) {
      cg_opts.dialect = Dialect::C99;
    }
  }
  std::vector<std::string> input_shapes(InputsShape.begin(), InputsShape.end());
  std::vector<std::string> inputs(Inputs.begin(), Inputs.end());
  std::vector<std::string> outputs(Outputs.begin(), Outputs.end());
  const auto& fusion_opts = GetFusionOptions();

  PopulateOptPasses(&pm, Target, input_shapes, inputs, outputs, Batch,
                    PreprocessScale, ReorderChannelLayout, SplitFunction,
                    DisableTypeCast, format, cg_opts, fusion_opts);
  PopulateCodeGenPasses(&pm, out_code, out_constants, out_header,
                        out_dynamic_check, Target, is_c_or_cxx_output,
                        is_binary_output, EmitDataAsC, EmitCodeOnly, EmitLLVMIR,
                        EmitTritonConfig, TritonConfigFile, QuantWeights,
                        PGQFile, RISCVOpt, cg_opts);

  auto status = pm.Run(&m);

  if (PrintAll) {
    pm.Dump();
    m.Dump();
  }

  if (status != Status::SUCCESS) {
    return -1;
  }

  if (!DisableCodeFormat && is_c_or_cxx_output && of_code.good()) {
    of_code.close();
    FormatCode(OutputFile);
  }
  if (!DisableCodeFormat && of_header.good()) {
    of_header.close();
    FormatCode(header_file_name.str());
  }
  return 0;
}
