//===- driver.cc ----------------------------------------------------------===//
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
#include <iterator>
#include <set>
#include <string>

#include "halo/halo.h"
#include "halo/lib/framework/common.h"
#include "halo/lib/ir/ir_builder.h"
#include "halo/lib/parser/parser.h"
#include "halo/lib/pass/pass_manager.h"
#include "halo/lib/transforms/fusion.h"
#include "halo/utils/cl_options.h"
#include "halo/utils/passes_helper.h"
#include "halo/utils/path.h"
#include "halo/version.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Support/CommandLine.h"

using namespace halo;

static llvm::cl::opt<std::string> Target("target",
                                         llvm::cl::desc("target triple"),
                                         llvm::cl::init("x86_64-unknown-linux"),
                                         llvm::cl::cat(HaloOptCat));
static llvm::cl::opt<std::string> Processor("processor",
                                            llvm::cl::desc("processor name"),
                                            llvm::cl::init("native"),
                                            llvm::cl::cat(HaloOptCat));
static llvm::cl::opt<std::string> OutputFile(
    "o", llvm::cl::desc("output file name."), llvm::cl::Required,
    llvm::cl::cat(HaloOptCat));

static llvm::cl::opt<bool> PrintAll(
    "print-all", llvm::cl::desc("print intermediates of all passes"),
    llvm::cl::init(false), llvm::cl::cat(HaloOptCat));

static llvm::cl::opt<bool> PrintPass("print-pass",
                                     llvm::cl::desc("print pass name"),
                                     llvm::cl::init(false),
                                     llvm::cl::cat(HaloOptCat));

static llvm::cl::opt<bool> EmitObj(
    "emit-obj", llvm::cl::desc("output the object code of ODLA"),
    llvm::cl::init(false), llvm::cl::cat(HaloOptCat));

static llvm::cl::opt<bool> EmitLibrary(
    "emit-lib", llvm::cl::desc("output the shared library"),
    llvm::cl::init(false), llvm::cl::cat(HaloOptCat));

static llvm::cl::opt<std::string> LinkODLALib(
    "link-odla-lib", llvm::cl::desc("link with ODLA library"),
    llvm::cl::init(""), llvm::cl::cat(HaloOptCat));

static llvm::cl::opt<bool> EmitLLVMIR("emit-llvm",
                                      llvm::cl::desc("output the LLVM IR code"),
                                      llvm::cl::init(false),
                                      llvm::cl::cat(HaloOptCat));

static llvm::cl::opt<std::string> ModuleName("module-name",
                                             llvm::cl::desc("name of module"),
                                             llvm::cl::init("halo_module"),
                                             llvm::cl::cat(HaloOptCat));

static llvm::cl::opt<ChannelOrder> ReorderChannelLayout(
    llvm::cl::values(clEnumValN(ChannelOrder::None, "none", "No reordering"),
                     clEnumValN(ChannelOrder::ChannelFirst, "channel-first",
                                "Reorder to channel first"),
                     clEnumValN(ChannelOrder::ChannelLast, "channel-last",
                                "Reorder to channel last")),
    "reorder-data-layout", llvm::cl::desc("Reorder the data layout"),
    llvm::cl::init(ChannelOrder::None), llvm::cl::cat(HaloOptCat));

static llvm::cl::opt<bool> RemoveInputTranspose(
    "remove-input-transpose", llvm::cl::desc("Remove the transpose for inputs"),
    llvm::cl::init(false), llvm::cl::cat(HaloOptCat));

static llvm::cl::opt<bool> RemoveOutputTranspose(
    "remove-output-transpose",
    llvm::cl::desc("Remove the transpose for outputs"), llvm::cl::init(false),
    llvm::cl::cat(HaloOptCat));

static llvm::cl::opt<bool> SeparateConstants(
    "separate-constants",
    llvm::cl::desc("Generate separate file for constants"),
    llvm::cl::init(true), llvm::cl::cat(HaloOptCat));
static llvm::cl::opt<bool> SimplifyForPreprocess(
    "simplify-for-preprocess",
    llvm::cl::desc("assume the input is preprocessed"), llvm::cl::init(false),
    llvm::cl::cat(HaloOptCat), llvm::cl::ReallyHidden);
static llvm::cl::opt<bool> DisableBroadcasting(
    "disable-broadcasting", llvm::cl::desc("disable broadcasting of constants"),
    llvm::cl::init(true), llvm::cl::cat(HaloOptCat));
static llvm::cl::opt<bool> DisableConvBN(
    "disable-convert-bn",
    llvm::cl::desc("disable convert Batch Normalization into mul/add"),
    llvm::cl::init(false), llvm::cl::cat(HaloOptCat));
static llvm::cl::opt<bool> FuseConvBias("fuse-conv-bias",
                                        llvm::cl::desc("fuse conv bias"),
                                        llvm::cl::init(false),
                                        llvm::cl::cat(HaloOptCat));
static llvm::cl::opt<bool> FuseMatMulMul(
    "fuse-matmul-mul", llvm::cl::desc("fuse matmul && mul layer"),
    llvm::cl::init(false), llvm::cl::cat(HaloOptCat));
static llvm::cl::opt<bool> FuseHSwish("fuse-h-swish",
                                      llvm::cl::desc("fuse h-swish"),
                                      llvm::cl::init(false),
                                      llvm::cl::cat(HaloOptCat));
static llvm::cl::opt<bool> FuseRelu("remove-relu-after-conv",
                                    llvm::cl::desc("remove relu after conv"),
                                    llvm::cl::init(false),
                                    llvm::cl::cat(HaloOptCat));
static llvm::cl::opt<bool> FuseFC("fuse-fully-connected",
                                  llvm::cl::desc("fuse to fully-connected"),
                                  llvm::cl::init(true),
                                  llvm::cl::cat(HaloOptCat));
static llvm::cl::opt<bool> FuseMultoConv("fuse-mul-to-conv",
                                         llvm::cl::desc("fuse mul to conv"),
                                         llvm::cl::init(true),
                                         llvm::cl::cat(HaloOptCat));
static llvm::cl::opt<bool> EmitCodeOnly(
    "code-only", llvm::cl::desc("Generate the code only"),
    llvm::cl::init(false), llvm::cl::cat(HaloOptCat));

static llvm::cl::opt<bool> ConvertSplitToSlice(
    "convert-split-to-slice", llvm::cl::desc("convert split to slice"),
    llvm::cl::init(true), llvm::cl::cat(HaloOptCat));

static llvm::cl::opt<bool> ConvertSquaredDifference(
    "convert-squared-diff",
    llvm::cl::desc("convert squaredDifference to sub/mul"),
    llvm::cl::init(true), llvm::cl::cat(HaloOptCat));

static llvm::cl::opt<bool> RISCVOpt(
    "riscv-opt", llvm::cl::desc("Enable optimizations for RISC-V only"),
    llvm::cl::init(false), llvm::cl::cat(HaloOptCat));

static llvm::cl::opt<BF16Mode> OptBF16Mode(
    llvm::cl::values(
        clEnumValN(BF16Mode::Disable, "disable", "disable bf16 mode"),
        clEnumValN(BF16Mode::Accuracy, "accuracy", "white list Model"),
        clEnumValN(BF16Mode::Performace, "performace",
                   "global enable bf16,except black list"),
        clEnumValN(BF16Mode::Auto, "auto", "automixprecision")),
    "bf16-mode", llvm::cl::desc("Enable BF16 with acc/perf/auto mode"),
    llvm::cl::init(BF16Mode::Disable), llvm::cl::cat(HaloOptCat));

static llvm::cl::opt<bool> EnableFP16("enable-fp16",
                                      llvm::cl::desc("Enable FP16 mode"),
                                      llvm::cl::init(false),
                                      llvm::cl::cat(HaloOptCat));

static llvm::cl::opt<bool> EnableIpuDevice("enable-ipu-device",
                                           llvm::cl::desc("Enable IPU Device"),
                                           llvm::cl::init(false),
                                           llvm::cl::cat(HaloOptCat));

static llvm::cl::opt<bool> UseIpuModel("use-ipu-model",
                                       llvm::cl::desc("Use IPU Mode"),
                                       llvm::cl::init(false),
                                       llvm::cl::cat(HaloOptCat));

static llvm::cl::opt<int> IpuNum(
    "ipu-num",
    llvm::cl::desc("Num of ipus, should consistent with subgraph num"),
    llvm::cl::init(1), llvm::cl::cat(HaloOptCat));

static llvm::cl::opt<int> BatchesPerStep(
    "batches-per-step", llvm::cl::desc("Specify batches num for each step"),
    llvm::cl::init(1), llvm::cl::cat(HaloOptCat));

static llvm::cl::opt<bool> DisableCodeFormat(
    "disable-code-format",
    llvm::cl::desc("Disable formatting the generated C/C++ code"),
    llvm::cl::init(false), llvm::cl::cat(HaloOptCat));

static llvm::cl::opt<ExecMode> OptExecMode(
    llvm::cl::values(
        clEnumValN(ExecMode::Compile, "compile", "Compilation-Execution Model"),
        clEnumValN(ExecMode::Interpret, "interpret", "Interpreter Model")),
    "exec-mode", llvm::cl::desc("Execution model of emitted code"),
    llvm::cl::init(ExecMode::Compile), llvm::cl::cat(HaloOptCat));

static llvm::cl::opt<bool> EmitDataAsC(
    "emit-data-as-c", llvm::cl::desc("Emit Constants as C/C++ code"),
    llvm::cl::init(false), llvm::cl::cat(HaloOptCat));

static llvm::cl::opt<bool> PrintMemStats(
    "print-mem-stats", llvm::cl::desc("Print Memory Usage Stats"),
    llvm::cl::init(false), llvm::cl::cat(HaloOptCat));

static llvm::cl::opt<bool> EmitValueReset(
    "emit-value-reset",
    llvm::cl::desc("Emit code to reset value life cycle ends"),
    llvm::cl::init(false), llvm::cl::cat(HaloOptCat));

static llvm::cl::opt<bool> EmitValueIDAsInt(
    "emit-value-id-as-int",
    llvm::cl::desc("Emit value id as integer. (default is string"),
    llvm::cl::init(false), llvm::cl::cat(HaloOptCat));

static llvm::cl::opt<bool> SplitFunction(
    "fiss-function",
    llvm::cl::desc("Split the function into multiple subfunctions"),
    llvm::cl::init(false), llvm::cl::cat(HaloOptCat));

static llvm::cl::opt<halo::API> Api(
    llvm::cl::values(
        clEnumValN(halo::API::HALO_RT, "halo_rt", "Using Halo Runtime Library"),
        clEnumValN(halo::API::ODLA_05, "odla_05", "Using ODLA 0.5")),
    "api", llvm::cl::desc("APIs used in emitted code"),
    llvm::cl::init(halo::API::ODLA_05), llvm::cl::cat(HaloOptCat));

static llvm::cl::opt<bool> EmitInferenceFunctionSignature(
    "emit-inference-func-sig",
    llvm::cl::desc("Emit fuction with a universal signature in c/c++ codegen"),
    llvm::cl::init(false), llvm::cl::cat(HaloOptCat));

static llvm::cl::opt<bool> EmitTritonConfig(
    "emit-triton-config",
    llvm::cl::desc("Emit triton inference server config file"),
    llvm::cl::init(false), llvm::cl::cat(HaloOptCat));

static llvm::cl::opt<std::string> TritonConfigFile(
    "triton-config-file", llvm::cl::desc("Triton inference server config file"),
    llvm::cl::init("config.pbtxt"), llvm::cl::cat(HaloOptCat));

static llvm::cl::list<std::string> Inputs(
    "inputs",
    llvm::cl::desc("Specify input names like -inputs=foo -inputs=bar"),
    llvm::cl::cat(HaloOptCat));

static llvm::cl::list<std::string> Outputs(
    "outputs",
    llvm::cl::desc("Specify output names like -outputs=foo, -outputs=bar:0"),
    llvm::cl::cat(HaloOptCat));

static llvm::cl::opt<Quantization> QuantWeights(
    llvm::cl::values(clEnumValN(Quantization::QUINT8, "quint8",
                                "Quantize weigths as quint8"),
                     clEnumValN(Quantization::FLOAT16, "float16",
                                "Quantize weigths as float16")),
    llvm::cl::values(clEnumValN(Quantization::QUINT16, "quint16",
                                "Quantize weigths as quint16")),
    "quantize-weights", llvm::cl::desc("Emit weights as quantized"),
    llvm::cl::init(Quantization::None), llvm::cl::cat(HaloOptCat));

static llvm::cl::opt<bool> DisableTypeCast(
    "disable-type-cast", llvm::cl::desc("Disable casting int64 to int32"),
    llvm::cl::init(true), llvm::cl::cat(HaloOptCat));

static llvm::cl::opt<signed> MaxBatch("max-batch-size",
                                      llvm::cl::desc("Specify max batch size"),
                                      llvm::cl::init(8),
                                      llvm::cl::cat(HaloOptCat));

static llvm::cl::opt<signed> MinBatch("min-batch-size",
                                      llvm::cl::desc("Specify min batch size"),
                                      llvm::cl::init(1),
                                      llvm::cl::cat(HaloOptCat));

static llvm::cl::opt<signed> OptBatch("opt-batch-size",
                                      llvm::cl::desc("Specify opt batch size"),
                                      llvm::cl::init(4),
                                      llvm::cl::cat(HaloOptCat));
static llvm::cl::opt<std::string> PGQFile(
    "pgq-file", llvm::cl::init(""),
    llvm::cl::desc("Profiling file for quantization of biases"),
    llvm::cl::cat(HaloOptCat));

static llvm::cl::opt<std::string> QuantTable(
    "quant-tbl", llvm::cl::init("quant_infos"),
    llvm::cl::desc("quant table name used for table gen"),
    llvm::cl::cat(HaloOptCat));

static llvm::cl::opt<bool> CheckModel("check-model",
                                      llvm::cl::desc("Dynamic check model"),
                                      llvm::cl::init(false),
                                      llvm::cl::cat(HaloOptCat));

static llvm::cl::list<std::string> IncludePaths(
    "I", llvm::cl::desc("Add directory to include search paths"),
    llvm::cl::cat(HaloOptCat));

static llvm::cl::list<std::string> LibPaths(
    "L", llvm::cl::desc("Add directory to library search paths"),
    llvm::cl::cat(HaloOptCat));

static llvm::cl::opt<bool> SaveTemps(
    "save-temps", llvm::cl::desc("Save intermediate compilation results"),
    llvm::cl::init(false), llvm::cl::cat(HaloOptCat));
static llvm::cl::opt<std::string> TemplateFile(
    "use-template-file", llvm::cl::init(""),
    llvm::cl::desc("Template file for ODLA code generation"),
    llvm::cl::cat(HaloOptCat));

static llvm::cl::opt<bool> ConstantDecombine(
    "constant-decombine", llvm::cl::desc("Constant Decombine"),
    llvm::cl::init(false), llvm::cl::cat(HaloOptCat));

#undef HALO_FUSION_OPTIONS
#define HALO_FUSION_CMD_OPTIONS_DECL
#include "halo/lib/ir/fusion.cc.inc"
static llvm::cl::opt<bool> FuseLayernorm("fuse-layernorm",
                                         llvm::cl::desc("fuse layernorm"),
                                         llvm::cl::init(true),
                                         llvm::cl::cat(HaloOptCat));
static llvm::cl::opt<bool> FuseGelu("fuse-gelu", llvm::cl::desc("fuse gelu"),
                                    llvm::cl::init(true),
                                    llvm::cl::cat(HaloOptCat));
static llvm::cl::opt<bool> FuseMHA("fuse-mha",
                                   llvm::cl::desc("fuse multi-head attention"),
                                   llvm::cl::init(false),
                                   llvm::cl::cat(HaloOptCat));
#undef HALO_FUSION_CMD_OPTIONS_DECL

static void PrintVersion(llvm::raw_ostream& os) {
  os << "  Version:\t" << HALO_VERSION_MAJOR << '.' << HALO_VERSION_MINOR << '.'
     << HALO_VERSION_PATCH << '\n';
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
  llvm::cl::HideUnrelatedOptions(HaloOptCat);
  llvm::cl::ParseCommandLineOptions(argc, argv);
  GlobalContext ctx;
  ctx.SetBasePath(GetBaseDir(argv[0]));
  ctx.SetODLAIncludePath(FindODLAIncPath(ctx.GetBasePath(), IncludePaths));
  ctx.SetODLALibraryPath(
      FindODLALibPath(ctx.GetBasePath(), LibPaths, LinkODLALib));
  ctx.SetTargetTriple(Target);
  ctx.SetProcessorName(Processor);
  ctx.SetPrintPass(PrintPass);

  Module m(ctx, ModuleName);

  armory::Opts opts;
  ModelFormat format = ModelFormat::INVALID;
  if (ParseModels(ModelFiles, Format, EntryFunctionName, opts, &m, &format) !=
      Status::SUCCESS) {
    return 1;
  }

  if (PrintAll) {
    m.Dump();
  }

  ctx.SetVerbosity(Verbose ? 1 : 0);
  PassManager pm(ctx);

  std::ostringstream buf_code;
  std::ostringstream buf_header;
  std::ostringstream buf_constants;
  std::ofstream of_dynamic_check;
  std::ostream* out_dynamic_check = &std::cout;

  bool is_binary_output = EmitObj;

  llvm::StringRef target_name(Target);
  bool is_c_or_cxx_output =
      target_name.startswith_lower("cxx") || target_name.startswith_lower("cc");
  if (EmitTritonConfig) {
    if (!TritonConfigFile.empty() &&
        llvm::sys::path::filename(TritonConfigFile).equals(TritonConfigFile)) {
      llvm::SmallString<128> file_name;
      llvm::sys::path::append(file_name,
                              llvm::sys::path::parent_path(OutputFile),
                              TritonConfigFile);
      TritonConfigFile = std::string(file_name);
    }
  }
  llvm::StringRef name(OutputFile);

  if (CheckModel) {
    of_dynamic_check.open(GetDerivedFileName(name.str(), ".main.cc.in"),
                          std::ofstream::binary);
    out_dynamic_check = &of_dynamic_check;
  }

  CXXCodeGenOpts cg_opts;
  cg_opts.simplify_for_preprocess = SimplifyForPreprocess;
  cg_opts.bf16_mode = OptBF16Mode;
  cg_opts.print_mem_stats = PrintMemStats;
  cg_opts.emit_value_reset = EmitValueReset;
  cg_opts.exec_mode = OptExecMode;
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
  cg_opts.enable_type_cast = !DisableTypeCast;
  cg_opts.separate_constants = SeparateConstants;
  cg_opts.disable_conv_bn = DisableConvBN;
  cg_opts.fuse_conv_bias = FuseConvBias;
  cg_opts.fuse_matmul_mul = FuseMatMulMul;
  cg_opts.fuse_hardswish = FuseHSwish;
  cg_opts.fuse_conv_relu = FuseRelu;
  cg_opts.fuse_fully_connected = FuseFC;
  cg_opts.fuse_mul_to_conv = FuseMultoConv;
  cg_opts.remove_input_transpose = RemoveInputTranspose;
  cg_opts.remove_output_transpose = RemoveOutputTranspose;
  cg_opts.emit_pb_file = target_name.startswith_lower("pb");
  cg_opts.format_code =
      !DisableCodeFormat && is_c_or_cxx_output && !is_binary_output;
  cg_opts.emit_header = true;
  cg_opts.emit_obj = EmitObj;
  cg_opts.emit_shared_lib = EmitLibrary || !LinkODLALib.empty();
  cg_opts.linked_odla_lib = LinkODLALib.c_str();
  cg_opts.save_temps = SaveTemps;
  cg_opts.constant_decombine = ConstantDecombine;
  cg_opts.quant_tbl = nullptr;
  if (QuantWeights != Quantization::None) {
    cg_opts.quant_tbl = QuantTable.c_str();
  }
  cg_opts.convert_split_to_slice = ConvertSplitToSlice;
  cg_opts.convert_squared_diff = ConvertSquaredDifference;

  if (!TemplateFile.empty()) {
    auto path = FindTemplateFile(ctx.GetBasePath(), TemplateFile);
    if (!path.empty()) {
      TemplateFile = path;
      cg_opts.template_file = TemplateFile.c_str();
    } else {
      cg_opts.template_file = nullptr;
      std::cerr << "Cannot find template file " << TemplateFile << std::endl;
    }
  }

  if (is_c_or_cxx_output) {
    if (llvm::StringRef(Target).startswith_lower("cc")) {
      cg_opts.dialect = Dialect::C99;
    }
    if (is_binary_output && ctx.GetODLAIncludePath().empty()) {
      llvm::errs() << "error: unable to find ODLA include path\n";
      return 1;
    }
    if (!LinkODLALib.empty() && ctx.GetODLALibraryPath().empty()) {
      llvm::errs() << "warning: unable to find ODLA library path\n";
    }
  }
  std::vector<std::string> input_shapes(InputsShape.begin(), InputsShape.end());
  std::vector<std::string> inputs(Inputs.begin(), Inputs.end());
  std::vector<std::string> outputs(Outputs.begin(), Outputs.end());
  auto fusion_opts = GetFusionOptions();
  fusion_opts.FuseLayerNorm = FuseLayernorm;
  fusion_opts.FuseGelu = FuseGelu;
  fusion_opts.FuseMHA = FuseMHA;

  is_binary_output = name.endswith(".bc");
  if (EmitObj.getNumOccurrences() == 0 && name.endswith(".o")) {
    cg_opts.emit_obj = true;
  }
  if (EmitLibrary.getNumOccurrences() == 0 && name.endswith(".so")) {
    cg_opts.emit_shared_lib = true;
  }
  cg_opts.channel_order = ReorderChannelLayout;
  PopulateOptPasses(&pm, Target, input_shapes, inputs, outputs, Batch,
                    PreprocessScale, SplitFunction, format, cg_opts,
                    fusion_opts);
  PopulateCodeGenPasses(&pm, &buf_code, &buf_constants, &buf_header,
                        out_dynamic_check, Target, is_c_or_cxx_output,
                        is_binary_output, EmitDataAsC, EmitCodeOnly, EmitLLVMIR,
                        EmitTritonConfig, TritonConfigFile, QuantWeights,
                        PGQFile, RISCVOpt, cg_opts, OutputFile);
  ModelFiles.removeArgument();
  auto status = pm.Run(&m);

  if (PrintAll) {
    pm.Dump();
    m.Dump();
  }

  if (status != Status::SUCCESS) {
    return -1;
  }

  if (!cg_opts.emit_shared_lib) {
    std::ofstream of_code;
    if (OutputFile == "-") {
      std::cout.rdbuf(of_code.rdbuf());
    } else {
      of_code.open(OutputFile, std::ofstream::binary);
    }
    of_code << buf_code.str();
  }

  if (cg_opts.emit_header) {
    llvm::SmallString<128> header_file_name("");
    header_file_name = name;
    llvm::sys::path::replace_extension(header_file_name, ".h");
    std::ofstream of_header;
    if (OutputFile == "-") {
      std::cout.rdbuf(of_header.rdbuf());
    } else {
      of_header.open(std::string(header_file_name));
    }
    of_header << buf_header.str();
  }

  if (!EmitCodeOnly && !cg_opts.emit_shared_lib) {
    llvm::StringRef name(OutputFile);
    llvm::SmallString<128> data_file_name(name);
    if (EmitDataAsC) {
      std::string ext =
          llvm::StringRef(Target).startswith_lower("cc") ? "data.c" : "data.cc";
      llvm::sys::path::replace_extension(data_file_name, ext);
    } else {
      llvm::sys::path::replace_extension(data_file_name, ".bin");
    }

    std::ofstream of_constants;
    of_constants.open(std::string(data_file_name), std::ofstream::binary);
    of_constants << buf_constants.str();
  }
  return 0;
}
