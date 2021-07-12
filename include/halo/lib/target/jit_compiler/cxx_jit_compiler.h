//===- cxx_jit_compiler.h ---------------------------------------*- C++ -*-===//
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

#ifndef HALO_LIB_TARGET_JIT_COMPILER_CXX_JIT_COMPILER_H_
#define HALO_LIB_TARGET_JIT_COMPILER_CXX_JIT_COMPILER_H_

#include <sstream>

#include "halo/halo.h"
#include "halo/lib/pass/pass.h"
#include "halo/lib/target/codegen.h"

namespace halo {

struct CXXCodeGenOpts;
// The class generates object code from c/cxx code.
class CXXJITCompiler final : public CodeGen {
 public:
  CXXJITCompiler(std::ostringstream& code, const std::ostringstream& source,
                 const std::vector<std::string>& header_searchs,
                 const CXXCodeGenOpts& opts)
      : CodeGen("CXX JIT Compiler"),
        source_(source),
        code_(code),
        header_searchs_(header_searchs),
        opts_(opts) {}
  virtual ~CXXJITCompiler() = default;

  bool RunOnModule(Module* module) override;

 private:
  const std::ostringstream& source_;
  std::ostringstream& code_;
  std::vector<std::string> header_searchs_;
  const CXXCodeGenOpts& opts_;
};

} // end namespace halo.

#endif // HALO_LIB_TARGET_JIT_COMPILER_CXX_JIT_COMPILER_H_