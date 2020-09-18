//===- compilation.h --------------------------------------------*- C++ -*-===//
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

#ifndef HALO_LIB_COMPILER_COMPILATION_H_
#define HALO_LIB_COMPILER_COMPILATION_H_

#include <memory>

namespace halo {

class GlobalContext;
class Module;

class Compilation {
 public:
  Compilation() = delete;
  explicit Compilation(GlobalContext& c);
  ~Compilation();

 private:
  std::unique_ptr<Module> module_ = nullptr;
};

} // namespace halo

#endif // HALO_LIB_COMPILER_COMPILATION_H_