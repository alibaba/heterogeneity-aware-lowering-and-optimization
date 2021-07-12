//===- code_formatter.h -----------------------------------------*- C++ -*-===//
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

#ifndef HALO_LIB_TARGET_GENERIC_CXX_CODE_FORMATTER_H_
#define HALO_LIB_TARGET_GENERIC_CXX_CODE_FORMATTER_H_

#include "halo/lib/pass/pass.h"

namespace halo {
struct CXXCodeGenOpts;
// The generic CXX compiler, which is a module pass.
class CodeFormatter : public ModulePass {
 public:
  CodeFormatter(std::ostringstream& os_code, std::ostringstream& os_header,
                const CXXCodeGenOpts& opts)
      : ModulePass("code format"),
        os_code_(os_code),
        os_header_(os_header),
        opts_(opts) {}

 protected:
  virtual bool RunOnModule(Module* m) override;

 private:
  std::ostringstream& os_code_;
  std::ostringstream& os_header_;
  const CXXCodeGenOpts& opts_;
};

} // end namespace halo.

#endif // HALO_LIB_TARGET_GENERIC_CXX_CODE_FORMATTER_H_
