//===- path.h ---------------------------------------------------*- C++ -*-===//
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
#ifndef HALO_UTILS_PATH_H_
#define HALO_UTILS_PATH_H_

#include <string>
#include <vector>

namespace halo {

std::string GetBaseDir(const char* argv0);
std::string GetBaseDir();

std::string FindODLAIncPath(const std::string& base_dir,
                            const std::vector<std::string>& include_paths);

std::string FindODLALibPath(const std::string& base_dir,
                            const std::vector<std::string>& lib_paths,
                            const std::string& libname);

std::string GetDerivedFileName(const std::string& main_file_name,
                               const std::string& ext);
} // namespace halo

#endif // HALO_UTILS_PATH_H_
