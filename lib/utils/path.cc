//===- path.cc ------------------------------------------------------------===//
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

#include "halo/utils/path.h"

#include <experimental/filesystem>
#include <string>

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

namespace halo {

static std::string GetAbsoluteDir(const std::string& dir) {
  if (llvm::sys::path::is_absolute(dir)) {
    return dir;
  }
  constexpr int buf_size = 128;
  llvm::SmallString<buf_size> curr_path;
  llvm::sys::fs::current_path(curr_path);
  return (llvm::StringRef(curr_path) + llvm::sys::path::get_separator() + dir)
      .str();
}

static std::string FindFirstOfFile(const std::vector<std::string>& dirs,
                                   const std::string& filename) {
  for (const auto& dir : dirs) {
    auto prefix = GetAbsoluteDir(dir);
    if (llvm::sys::fs::exists(prefix + llvm::sys::path::get_separator() +
                              filename)) {
      return prefix;
    }
  }
  return "";
}

std::string GetBaseDir(const char* argv0) {
  auto dir = llvm::sys::path::parent_path(argv0);
  std::string exe_dir = GetAbsoluteDir(dir.str());
  // assume the halo is under "bin" directory.
  return llvm::sys::path::parent_path(exe_dir).str();
}

std::string GetBaseDir() {
  const auto& curr = std::experimental::filesystem::current_path();
  return curr;
}

std::string FindODLAIncPath(const std::string& base_dir,
                            const std::vector<std::string>& include_paths) {
  std::vector<std::string> search_dirs = include_paths;
  search_dirs.push_back(base_dir + "/ODLA/include");
  search_dirs.push_back("/usr/include");
  search_dirs.push_back("/usr/local/include");
  search_dirs.push_back("/opt/halo/include");

  return FindFirstOfFile(search_dirs, "ODLA/odla.h");
}

std::string FindODLALibPath(const std::string& base_dir,
                            const std::vector<std::string>& lib_paths,
                            const std::string& libname) {
  std::vector<std::string> search_dirs = lib_paths;
  search_dirs.push_back(base_dir + "/lib");
  search_dirs.push_back("/usr/lib");
  search_dirs.push_back("/usr/local/lib");
  search_dirs.push_back("/opt/halo/lib");

  auto ret = FindFirstOfFile(
      search_dirs, libname.empty() ? "libodla_profiler.so" : libname + ".so");
  if (ret.empty()) {
    ret = FindFirstOfFile(search_dirs, "libodla_" + libname + ".so");
  }
  if (ret.empty()) {
    ret = FindFirstOfFile(search_dirs, "lib" + libname + ".so");
  }
  return ret;
}

std::string GetDerivedFileName(const std::string& main_file_name,
                               const std::string& ext) {
  constexpr int buf_len = 128;
  llvm::SmallString<buf_len> filename(main_file_name);
  llvm::sys::path::replace_extension(filename, ext);
  return filename.str().str();
}

} // namespace halo
