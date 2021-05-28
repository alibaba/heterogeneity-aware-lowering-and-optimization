//===- cxx_jit_compiler.cc ------------------------------------------------===//
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

#include "halo/lib/target/jit_compiler/cxx_jit_compiler.h"

#include <clang/CodeGen/CodeGenAction.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Tooling/CompilationDatabase.h>
#include <clang/Tooling/Tooling.h>
#include <llvm/Support/TargetRegistry.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_os_ostream.h>

#include <fstream>

#include "halo/halo.h"

class CodeGenActionFactory : public clang::tooling::FrontendActionFactory {
 public:
  CodeGenActionFactory() : os_{nullptr} {};
  explicit CodeGenActionFactory(llvm::raw_ostream& oss) {
    os_ = std::make_unique<llvm::buffer_ostream>(oss);
  }
  bool runInvocation(
      std::shared_ptr<clang::CompilerInvocation> invocation,
      clang::FileManager* files,
      std::shared_ptr<clang::PCHContainerOperations> pch_container_ops,
      clang::DiagnosticConsumer* diag_consumer) override {
    clang::CompilerInstance compiler(std::move(pch_container_ops));
    compiler.setInvocation(std::move(invocation));
    compiler.setFileManager(files);
    compiler.setOutputStream(std::move(os_));
    std::unique_ptr<clang::FrontendAction> scoped_tool_action(create());

    compiler.createDiagnostics(diag_consumer, /*ShouldOwnClient=*/false);

    compiler.createSourceManager(*files);

    const bool success = compiler.ExecuteAction(*scoped_tool_action);

    files->clearStatCache();
    return success;
  }

  clang::FrontendAction* create() override {
    return new clang::EmitObjAction();
  }

 private:
  std::unique_ptr<llvm::raw_pwrite_stream> os_;
};

class HaloCompilationDatabase
    : public clang::tooling::FixedCompilationDatabase {
 public:
  HaloCompilationDatabase() : FixedCompilationDatabase("/", {}) {}
  std::vector<clang::tooling::CompileCommand> getCompileCommands(
      llvm::StringRef file_path) const override {
    auto tmp =
        clang::tooling::FixedCompilationDatabase::getCompileCommands(file_path);
    for (auto& t : tmp) {
      t.CommandLine[0] = "halo-jit";
    }
    return tmp;
  }
};

static void EmitObj(std::ostream& output, const std::string& code,
                    const std::vector<std::string>& inc_dirs, bool is_cxx,
                    bool verbose, const std::vector<std::string>& extra_ops) {
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  HaloCompilationDatabase comp_db;
  const std::string filename = is_cxx ? "/input.cpp" : "/input.c";
  clang::tooling::ClangTool tool(comp_db, {filename});
  tool.mapVirtualFile(filename, code);
  tool.clearArgumentsAdjusters();
  // use list to avoid buffer reallocation.
  std::list<std::string> opts;
  for (const auto& inc_dir : inc_dirs) {
    opts.push_back("-I" + (inc_dir.empty() ? "." : inc_dir));
    tool.appendArgumentsAdjuster(
        clang::tooling::getInsertArgumentAdjuster(opts.back().c_str()));
  }

  if (verbose) {
    tool.appendArgumentsAdjuster(
        clang::tooling::getInsertArgumentAdjuster("-v"));
  }
  tool.appendArgumentsAdjuster(clang::tooling::getInsertArgumentAdjuster("-c"));
  tool.appendArgumentsAdjuster(
      clang::tooling::getInsertArgumentAdjuster("-O2"));
  tool.appendArgumentsAdjuster(
      clang::tooling::getInsertArgumentAdjuster("-fPIE"));
  for (const auto& opt : extra_ops) {
    tool.appendArgumentsAdjuster(
        clang::tooling::getInsertArgumentAdjuster(opt.c_str()));
  }
  tool.appendArgumentsAdjuster(clang::tooling::getInsertArgumentAdjuster("-o"));
  tool.appendArgumentsAdjuster(
      clang::tooling::getInsertArgumentAdjuster("/dummy.o"));
  llvm::raw_os_ostream os(output);
  auto factory = std::make_unique<CodeGenActionFactory>(os);
  tool.run(factory.get());
}

bool halo::CXXJITCompiler::RunOnModule(Module* module) {
  const auto& ctx = module->GetGlobalContext();
  const std::string& source = source_.str();
  if (opts_.save_temps) {
    constexpr int len = 128;
    llvm::SmallString<len> c_file;
    llvm::sys::fs::createTemporaryFile("halo_jit" /* prefix */, "c", c_file);
    std::cerr << "HALO intermediate ODLA file: " << c_file.str().str() << "\n";
    std::ofstream ofs(c_file.str(), std::ofstream::binary);
    ofs << source;
  }

  std::ostringstream buf;
  EmitObj(buf, source, {ctx.GetODLAIncludePath()},
          opts_.dialect == Dialect::CXX_11, ctx.GetVerbosity() > 0, {});
  buf.swap(code_);
  return false;
}
