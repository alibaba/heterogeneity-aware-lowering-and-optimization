//===- code_formatter.cc --------------------------------------------------===//
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

#include "halo/lib/target/generic_cxx/code_formatter.h"

#include <clang/Format/Format.h>
#include <clang/Frontend/FrontendActions.h>
#include <clang/Rewrite/Core/Rewriter.h>
#include <clang/Rewrite/Frontend/Rewriters.h>
#include <clang/Tooling/Core/Replacement.h>
#include <clang/Tooling/Tooling.h>
#include <halo/lib/framework/global_context.h>
#include <llvm/ADT/IntrusiveRefCntPtr.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/Path.h>
#include <llvm/Support/VirtualFileSystem.h>
#include <llvm/Support/raw_os_ostream.h>
#include <llvm/Support/raw_ostream.h>

#include <cstddef>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>

#include "halo/halo.h"
static bool FormatCode(std::ostringstream& input, bool is_cxx) {
  const std::string& code = input.str();
  unsigned int len = code.size();
  if (len == 0) {
    return true;
  }
  std::vector<clang::tooling::Range> ranges{{0, len}};
  auto code_buf = llvm::MemoryBuffer::getMemBuffer(code);

  std::string assumed_file_name = is_cxx ? "a.cc" : "a.c";
  auto style = clang::format::getLLVMStyle();

  style.SortIncludes = false;
  unsigned cursor_position = 0;
  auto replaces = clang::format::sortIncludes(
      style, code, ranges, assumed_file_name, &cursor_position);
  auto changed_code = clang::tooling::applyAllReplacements(code, replaces);
  if (!changed_code) {
    llvm::errs() << llvm::toString(changed_code.takeError()) << "\n";
    return false;
  }
  ranges = clang::tooling::calculateRangesAfterReplacements(replaces, ranges);
  clang::format::FormattingAttemptStatus status;
  auto format_changes =
      reformat(style, *changed_code, ranges, assumed_file_name, &status);
  replaces = replaces.merge(format_changes);

  llvm::IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> vfs(
      new llvm::vfs::InMemoryFileSystem(true));
  clang::FileManager files(clang::FileSystemOptions(), vfs);
  clang::DiagnosticsEngine diag(
      llvm::IntrusiveRefCntPtr<clang::DiagnosticIDs>(new clang::DiagnosticIDs),
      new clang::DiagnosticOptions);
  clang::SourceManager source_mgr(diag, files);
  vfs->addFileNoOwn(assumed_file_name, 0, code_buf.get());
  clang::FileID id =
      source_mgr.createFileID(files.getFile(assumed_file_name),
                              clang::SourceLocation(), clang::SrcMgr::C_User);
  clang::Rewriter rewriter(source_mgr, clang::LangOptions());
  clang::tooling::applyAllReplacements(replaces, rewriter);
  std::ostringstream buf;
  llvm::raw_os_ostream output(buf);
  rewriter.getEditBuffer(id).write(output);
  output.flush();
  input.swap(buf);
  return true;
}

namespace halo {

bool CodeFormatter::RunOnModule(Module* m) {
  bool is_cxx = opts_.dialect == Dialect::CXX_11;
  FormatCode(os_code_, is_cxx);
  FormatCode(os_header_, is_cxx);

  return false;
}

} // namespace halo
