//===- generic_jit_linker.cc ----------------------------------------------===//
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

#include "halo/lib/target/jit_compiler/generic_jit_linker.h"

#include <lld/Common/Driver.h>
#include <llvm/ADT/SmallString.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/FileSystem.h>

#include <fstream>

#include "halo/halo.h"

static void Link(const std::string& output_file,
                 const std::vector<std::string>& input_files,
                 const std::vector<std::string>& lib_search_paths,
                 const std::vector<std::string>& libs, bool shared,
                 bool verbose) {
  const char* prog = "halo-linker";
  const char* lib_search_opt = "-L";
  const char* lib_opt = "-l";
  const char* output_opt = "-o";
  const char* rpath_opt = "-R";
  const char* shared_opt = "--shared";
  std::vector<const char*> args{prog, shared_opt, output_opt,
                                output_file.c_str()};
  for (auto& f : input_files) {
    args.push_back(f.c_str());
  }
  for (auto& path : lib_search_paths) {
    if (!path.empty()) {
      args.push_back(lib_search_opt);
      args.push_back(path.c_str());
      args.push_back(rpath_opt);
      args.push_back(path.c_str());
    }
  }
  for (auto& lib : libs) {
    if (!lib.empty()) {
      args.push_back(lib_opt);
      args.push_back(lib.c_str());
    }
  }
  if (verbose) {
    for (auto& arg : args) {
      std::cerr << arg << " ";
    }
    std::cerr << "\n";
  }
  lld::elf::link(args, false);
}

static std::string WriteToTempFile(const std::ostringstream& data) {
  constexpr int len = 128;
  llvm::SmallString<len> obj_file;
  llvm::sys::fs::createTemporaryFile("halo_jit" /* prefix */, "o", obj_file);
  std::ofstream ofs;
  ofs.open(obj_file.str(), std::ofstream::binary);
  ofs << data.str();
  ofs.close();
  return obj_file.str();
}

bool halo::GenericJITLinker::RunOnModule(Module* module) {
  auto code_file_name = WriteToTempFile(obj_code_);
  auto constants_file_name = WriteToTempFile(obj_constants_);
  auto& ctx = module->GetGlobalContext();
  std::vector<std::string> libs;
  if (opts_.linked_odla_lib != nullptr) {
    libs.push_back(opts_.linked_odla_lib);
  }
  Link(output_file_name_, {code_file_name, constants_file_name},
       {ctx.GetODLALibraryPath()}, libs, true /* shared */,
       ctx.GetVerbosity() > 0);
  // llvm::sys::fs::remove(code_file_name);
  // llvm::sys::fs::remove(constants_file_name);
  return false;
}
