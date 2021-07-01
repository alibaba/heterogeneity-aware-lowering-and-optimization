//===- templated_cxx_codegen.h ----------------------------------*- C++ -*-===//
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

#ifndef HALO_LIB_TARGET_GENERIC_CXX_TEMPLATED_CXX_CODEGEN_H_
#define HALO_LIB_TARGET_GENERIC_CXX_TEMPLATED_CXX_CODEGEN_H_

#include "halo/lib/target/generic_cxx/generic_cxx_codegen.h"

namespace halo {

// The generic CXX compiler, which is a module pass.
class TemplatedCXXCodeGen : public GenericCXXCodeGen {
 public:
  TemplatedCXXCodeGen(std::ostringstream& os, std::ostringstream& header_os,
                      const CXXCodeGenOpts& opts);

  virtual ~TemplatedCXXCodeGen();

 protected:
  void RunOnFunction(Function& function) override;

 private:
  std::ostringstream generic_os_;
  std::ostream& code_os_;
};

} // end namespace halo.

#endif // HALO_LIB_TARGET_GENERIC_CXX_TEMPLATED_CXX_CODEGEN_H_
