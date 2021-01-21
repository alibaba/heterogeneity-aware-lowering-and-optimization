//===- codegen.cc ---------------------------------------------------------===//
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

#include "halo/lib/target/codegen.h"

#include <set>

#include "halo/lib/ir/all_instructions.h"

namespace halo {

void CodeGen::RunOnBaseInstruction(Instruction* inst) {
  switch (inst->GetOpCode()) {
#define GET_INST_DOWNCAST_SWITCH
#include "halo/lib/ir/instructions_info.def"
#undef GET_INST_DOWNCAST_SWITCH
    default: {
      HLCHECK(0 && "Unreachable");
    }
  }
}

const std::string& CodeGen::GetRTLibFuncName(const Instruction& inst) {
  auto kv = RuntimeLibFuncNames.find(inst.GetOpCode());
  HLCHECK(kv != RuntimeLibFuncNames.end());
  return kv->second;
}

std::string CodeGen::NormalizeVariableName(const std::string& name) {
  std::string ret(name);
  static std::set<std::string> keywords{"asm",          "auto",
                                        "bool",         "break",
                                        "case",         "catch",
                                        "char",         "const",
                                        "continue",     "do",
                                        "default",      "delete",
                                        "do",           "double",
                                        "dynamic_cast", "else",
                                        "enum",         "explicit",
                                        "false",        "float",
                                        "for",          "friend",
                                        "goto",         "if",
                                        "inline",       "int",
                                        "long",         "namespace",
                                        "new",          "private",
                                        "protected",    "public",
                                        "register",     "reinterpret_cast",
                                        "return",       "short",
                                        "signed",       "sizeof",
                                        "static",       "std",
                                        "switch",       "template",
                                        "this",         "throw",
                                        "true",         "try",
                                        "typedef",      "typeid",
                                        "typename",     "union",
                                        "unsigned",     "using",
                                        "virtual",      "void",
                                        "volatile",     "while"};
  std::transform(name.begin(), name.end(), ret.begin(), [](char c) {
    switch (c) {
      case '\'':
      case '/':
      case ' ':
      case '.':
      case '>':
      case '<':
      case '-': {
        return '_';
      }
      default:
        return c;
    }
  });
  if (std::isdigit(name[0]) != 0) {
    ret = "val_" + ret;
  }
  if (keywords.count(ret) != 0) {
    ret = "v_" + ret;
  }
  return ret;
}

} // namespace halo
