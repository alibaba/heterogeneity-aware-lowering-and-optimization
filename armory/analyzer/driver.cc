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
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

using namespace halo;

static void PopulatePassesAndRun(GlobalContext& ctx, Module& m,
                                 const llvm::cl::opt<signed>& batch,
                                 Parser::Format format) {
  PassManager pm(ctx);
  std::vector<std::string> input_shapes(InputsShape.begin(), InputsShape.end());
  pm.AddPass<InputLegalizer>(batch.getValue(), input_shapes);
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

int main(int argc, char** argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv);
  GlobalContext ctx;
  ctx.SetBasePath(argv[0]);

  Module m(ctx, "analyzer_module");

  armory::Opts opts;
  opts.convert_to_ipu_graphdef = ConvertToIpuGraphDef;
  opts.output_graphdef_filename = OutputGraphDefFile;
  Parser::Format format = Parser::Format::INVALID;
  if (ParseModels(ModelFiles, ModelFormat, EntryFunctionName, opts, &m,
                  &format) != Status::SUCCESS) {
    return 1;
  }

  PopulatePassesAndRun(ctx, m, Batch, format);
  return 0;
}
