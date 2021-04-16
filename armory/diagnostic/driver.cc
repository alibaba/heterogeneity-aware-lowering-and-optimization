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
#include "halo/utils/cl_options.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

using namespace halo;

static llvm::cl::opt<bool> PrintDiagnosticReport(
    "print-diagnostic-report", llvm::cl::desc("Print diagnostic report"),
    llvm::cl::init(false));

int main(int argc, char** argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv);
  GlobalContext ctx;
  ctx.SetBasePath(argv[0]);

  Module m(ctx, "diagnostic_module");

  armory::Opts opts(PrintDiagnosticReport);
  Parser::Format format = Parser::Format::INVALID;
  if (ParseModels(ModelFiles, ModelFormat, EntryFunctionName, opts, &m,
                  &format) != Status::SUCCESS) {
    return 1;
  }

  return 0;
}
