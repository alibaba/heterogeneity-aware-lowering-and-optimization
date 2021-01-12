//===- generic_cxx_constant_writer.cc -------------------------------------===//
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

#include "halo/lib/target/generic_cxx/generic_cxx_codegen.h"

namespace halo {

GenericCXXConstantWriter::GenericCXXConstantWriter(std::ostream& os)
    : GenericCXXCodeGen(/*"Generic CXX Constant Writer"*/ os, os) {}

void GenericCXXConstantWriter::RunOnConstant(const Constant& constant,
                                             std::ostream* os) {
  const auto& type = constant.GetResultType();
  CXXValue value(constant.GetName(),
                 GenericCXXCodeGen::TensorTypeToCXXType(type, true));
  // "extern" is needed for CXX compiler to prevent name mangling.
  *os << "extern const " << value.type.name << " " << value.name << "["
      << Join(type.GetDimSizes(), '*') << "] = {";
  constant.PrintData(os, constant.GetResultType().GetTotalNumOfElements());
  *os << "};\n";
}

bool GenericCXXConstantWriter::RunOnModule(Module* module) {
  os_ << "//===- Halo Compiler Generated File "
         "---------------------------------------===//\n\n";
  os_ << "#include <stdint.h>\n";

  for (auto& c : module->Constants()) {
    RunOnConstant(*c, &os_);
  }
  for (auto& func : *module) {
    for (auto& constant : func->Constants()) {
      RunOnConstant(*constant, &os_);
    }
  }

  return false;
}

} // namespace halo