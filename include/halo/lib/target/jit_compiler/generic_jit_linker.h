//===- generic_jit_linker.h -------------------------------------*- C++ -*-===//
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

#ifndef HALO_LIB_TARGET_JIT_COMPILER_GENERIC_JIT_COMPILER_H_
#define HALO_LIB_TARGET_JIT_COMPILER_GENERIC_JIT_COMPILER_H_

#include <sstream>

#include "halo/halo.h"
#include "halo/lib/target/codegen.h"

namespace halo {

struct CXXCodeGenOpts;
// The class generates linked library from object code.
class GenericJITLinker final : public CodeGen {
 public:
  GenericJITLinker(const std::ostringstream& obj_code,
                   const std::ostringstream& obj_constants,
                   const std::string& output_file_name,
                   const CXXCodeGenOpts& opts)
      : CodeGen("Generic JIT Linker"),
        obj_code_(obj_code),
        obj_constants_(obj_constants),
        output_file_name_(output_file_name),
        opts_(opts) {}

  bool RunOnModule(Module* module) override;

 private:
  const std::ostringstream& obj_code_;
  const std::ostringstream& obj_constants_;
  const std::string output_file_name_;
  const CXXCodeGenOpts& opts_;
};

} // end namespace halo.

#endif // HALO_LIB_TARGET_JIT_COMPILER_Generic_JIT_LINKER_H_