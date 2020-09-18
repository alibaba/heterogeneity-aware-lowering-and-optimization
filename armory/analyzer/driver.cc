//===- analyzer.cc --------------------------------------------------------===//
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
#include "halo/lib/transforms/analyzer.h"
#include "halo/lib/transforms/caffeextension_legalizer.h"
#include "halo/lib/transforms/dce.h"
#include "halo/lib/transforms/input_legalizer.h"
#include "halo/lib/transforms/inst_simplify.h"
#include "halo/lib/transforms/onnxextension_legalizer.h"
#include "halo/lib/transforms/tfextension_legalizer.h"
#include "halo/lib/transforms/type_legalizer.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

using namespace halo;

static llvm::cl::list<std::string> ModelFiles(
    llvm::cl::Positional, llvm::cl::desc("model file name."),
    llvm::cl::OneOrMore);

static llvm::cl::opt<Parser::Format> ModelFormat(
    "x",
    llvm::cl::desc(
        "format of the following input model files. Permissible formats "
        "include: TENSORFLOW CAFFE ONNX MXNET. If unspecified, the format is "
        "guessed base on file's extension."),
    llvm::cl::init(Parser::Format::INVALID));

static llvm::cl::opt<unsigned> Batch(
    "batch-size",
    llvm::cl::desc("Specify batch size if the first dim of input is negative"),
    llvm::cl::init(1));

static llvm::cl::opt<std::string> EntryFunctionName(
    "entry-func-name", llvm::cl::desc("name of entry function"),
    llvm::cl::init(""));

static llvm::cl::opt<bool> PrintAnalysisReport(
    "print-analysis-report", llvm::cl::desc("Print analysis report"),
    llvm::cl::init(false));

static void PopulatePassesAndRun(GlobalContext& ctx, Module& m,
                                 const llvm::cl::opt<unsigned>& batch,
                                 Parser::Format format) {
  PassManager pm(ctx);
  pm.AddPass<InputLegalizer>(batch.getValue(), std::vector<std::string>{});
  if (format == Parser::Format::CAFFE) {
    pm.AddPass<CAFFEExtensionLegalizer>();
  } else if (format == Parser::Format::TENSORFLOW) {
    pm.AddPass<TFExtensionLegalizer>();
  } else {
    HLCHECK(format == Parser::Format::ONNX);
    pm.AddPass<ONNXExtensionLegalizer>();
  }
  pm.AddPass<DCE>();
  pm.AddPass<TypeLegalizer>(true);
  auto analyzer = pm.AddPass<Analyzer>();
  pm.Run(&m);
  if (PrintAnalysisReport) {
    analyzer->WriteCSVReport(std::cout);
  }
}

/// Guess the model format based on input file extension.gg
static Parser::Format InferFormat(
    const llvm::cl::list<std::string>& model_files, size_t file_idx) {
  llvm::StringRef ext = llvm::sys::path::extension(model_files[file_idx]);
  auto format = llvm::StringSwitch<Parser::Format>(ext)
                    .Case(".pb", Parser::Format::TENSORFLOW)
                    .Case(".pbtxt", Parser::Format::TENSORFLOW)
                    .Case(".prototxt", Parser::Format::TENSORFLOW)
                    .Case(".onnx", Parser::Format::ONNX)
                    .Case(".json", Parser::Format::MXNET)
                    .Default(Parser::Format::INVALID);
  // Check the next input file to see if it is caffe.
  if (format == Parser::Format::TENSORFLOW &&
      (file_idx + 1 < model_files.size()) &&
      llvm::sys::path::extension(model_files[file_idx + 1]) == ".caffemodel") {
    format = Parser::Format::CAFFE;
  }
  return format;
}

static Status ParseModels(const llvm::cl::list<std::string>& model_files,
                          const llvm::cl::opt<Parser::Format>& model_format,
                          const llvm::cl::opt<std::string>& entry_func_name,
                          const armory::Opts& opts, Module* module,
                          Parser::Format* f) {
  std::set<std::string> func_names;
  for (size_t i = 0, e = model_files.size(); i < e; ++i) {
    Parser::Format format = model_format;
    if (format == Parser::Format::INVALID) {
      format = InferFormat(model_files, i);
    }
    HLCHECK(format != Parser::Format::INVALID);
    *f = format;
    FunctionBuilder func_builder(module);
    // Use stem of the input model as function name.
    std::string func_name = entry_func_name.empty()
                                ? llvm::sys::path::stem(model_files[i]).str()
                                : entry_func_name.getValue();
    while (func_names.count(func_name) != 0) {
      func_name.append("_").append(std::to_string(i));
    }
    func_names.insert(func_name);
    Function* func = func_builder.CreateFunction(func_name);
    std::vector<std::string> files{model_files[i]};
    if (format == Parser::Format::CAFFE || format == Parser::Format::MXNET) {
      HLCHECK(i + 1 < e);
      files.push_back(model_files[++i]);
    }
    if (Status status = Parser::Parse(func, format, files, opts);
        status != Status::SUCCESS) {
      return status;
    }
  }
  return Status::SUCCESS;
}

int main(int argc, char** argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv);
  GlobalContext ctx;
  ctx.SetBasePath(argv[0]);

  Module m(ctx, "analyzer_module");

  armory::Opts opts;
  Parser::Format format = Parser::Format::INVALID;
  if (ParseModels(ModelFiles, ModelFormat, EntryFunctionName, opts, &m,
                  &format) != Status::SUCCESS) {
    return 1;
  }

  PopulatePassesAndRun(ctx, m, Batch, format);
  return 0;
}