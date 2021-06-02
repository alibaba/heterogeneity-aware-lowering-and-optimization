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

#include "halo/halo.h"
#include "halo/lib/framework/common.h"
#include "halo/lib/ir/ir_builder.h"
#include "halo/lib/parser/parser.h"
#include "halo/lib/pass/pass_manager.h"
#include "halo/utils/cl_options.h"
#include "halo/utils/passes_helper.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

using namespace halo;

static llvm::cl::opt<bool> PrintAnalysisReport(
    "print-analysis-report", llvm::cl::desc("Print analysis report"),
    llvm::cl::init(false), llvm::cl::cat(HaloOptCat));

static llvm::cl::opt<bool> ConvertToIpuGraphDef(
    "convert-to-ipu-graphdef", llvm::cl::desc("Convert to IPU style graphdef"),
    llvm::cl::init(false), llvm::cl::cat(HaloOptCat));

static llvm::cl::opt<std::string> SplitNames(
    "split-names", llvm::cl::desc("Split names."),
    llvm::cl::desc(
        "Specify split names like -split-name=xxx:yyy,aaa:bbb,mmm:nnn"),
    llvm::cl::cat(HaloOptCat));

static llvm::cl::opt<std::string> OutputGraphDefFile(
    "graphdef-file-name", llvm::cl::desc("output graphdef file name."),
    llvm::cl::init("./converted_model.pb"), llvm::cl::cat(HaloOptCat));

static void PopulatePassesAndRun(GlobalContext& ctx, Module& m,
                                 const llvm::cl::opt<signed>& batch,
                                 ModelFormat format) {
  PassManager pm(ctx);
  std::vector<std::string> input_shapes(InputsShape.begin(), InputsShape.end());
  FusionOptions fusion_opts;
  CXXCodeGenOpts opts;
  PopulateOptPasses(&pm, "cxx", input_shapes, {}, {}, batch, "", false, false,
                    format, opts, fusion_opts);
  pm.AddAnalyzerPass(&std::cout);
  pm.Run(&m);
}

int main(int argc, char** argv) {
  llvm::cl::HideUnrelatedOptions(HaloOptCat);
  llvm::cl::ParseCommandLineOptions(argc, argv);
  GlobalContext ctx;
  ctx.SetBasePath(argv[0]);

  Module m(ctx, "analyzer_module");

  armory::Opts opts;
  opts.convert_to_ipu_graphdef = ConvertToIpuGraphDef;
  opts.output_graphdef_filename = OutputGraphDefFile.getValue();

  llvm::SmallVector<llvm::StringRef, 4> splitted;
  llvm::StringRef(SplitNames).split(splitted, ',');
  opts.split_names.reserve(splitted.size());
  for (size_t i = 0; i < splitted.size(); ++i) {
    if (!splitted[i].empty()) {
      llvm::SmallVector<llvm::StringRef, 4> name;
      llvm::StringRef(splitted[i]).split(name, ':');
      opts.split_names.emplace_back();
      for (auto& n : name) {
        if (!n.empty()) {
          opts.split_names[i].push_back(n.str());
        }
      }
    }
  }

  ModelFormat format = ModelFormat::INVALID;
  if (ParseModels(ModelFiles, Format, EntryFunctionName, opts, &m, &format) !=
      Status::SUCCESS) {
    return 1;
  }

  PopulatePassesAndRun(ctx, m, Batch, format);
  return 0;
}
