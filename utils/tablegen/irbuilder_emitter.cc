//===- inst_builder_emitter.cc ----------------------------------*- C++ -*-===//
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

#include <algorithm>

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

namespace halo {

static std::vector<llvm::Record*> GetAllInstDefs(
    const llvm::RecordKeeper& records) {
  std::vector<llvm::Record*> insts = records.getAllDerivedDefinitions("Inst");
  std::sort(insts.begin(), insts.end(),
            [](llvm::Record* lhs, llvm::Record* rhs) {
              auto op1 = lhs->getName();
              auto op2 = rhs->getName();
              return (op1.compare(op2) <= 0);
            });
  return insts;
}

/// Emit the function prototype. If `num_of_ops` is negative, emits the
/// operator argument as `const std::vector<Def>&`.
static llvm::raw_ostream& EmitFuncProto(llvm::raw_ostream& os,
                                        llvm::StringRef prefix,
                                        llvm::StringRef name, int num_of_ops,
                                        bool create_only) {
  if (create_only) {
    os << name << "Inst* " << prefix << "Create";
  } else {
    os << name << "Inst* " << prefix << "CreateAndAppend";
  }
  os << name << "(const std::string& name";
  if (num_of_ops < 0) {
    os << ", const std::vector<Def>& ops";
  }
  for (int i = 0; i < num_of_ops; ++i) {
    os << ", const Def& op" << i;
  }
  os << ")";
  return os;
}

static llvm::raw_ostream& EmitFuncBody(llvm::raw_ostream& os,
                                       llvm::StringRef name, int num_of_ops,
                                       bool create_only) {
  llvm::Twine class_name(name + "Inst");
  os << "  auto inst = std::make_unique<" << class_name
     << ">(GetContext(), name";
  for (int i = 0; i < num_of_ops; ++i) {
    os << ", op" << i;
  }
  if (num_of_ops < 0) {
    os << ", ops";
  }
  os << ");\n";
  os << "  inst->parent_basic_block_ = GetParent();\n";
  os << "  " << class_name << "*ret = inst.get();\n";
  if (create_only) {
    os << "  Insert(std::move(inst));\n";
  } else {
    os << "  Append(std::move(inst));\n";
  }
  os << "  return ret;\n";
  os << "}\n\n";
  return os;
}

/// Emits the creators of IR like
/// std::unique_ptr<XXXInst> CreateXXX(Def& op, ..., );
void EmitIRBuilder(const llvm::RecordKeeper& records, llvm::raw_ostream& os,
                   bool decl) {
  auto insts = GetAllInstDefs(records);
  llvm::StringRef prefix = decl ? "" : "IRBuilder::";
  llvm::StringRef line_end = decl ? ";\n" : " {\n";
  for (auto& inst : insts) {
    llvm::StringRef inst_name = inst->getName();
    int arg_min = inst->getValueAsListOfDefs("ins_").size();
    int arg_max = arg_min;
    for (int arg = arg_min; arg <= arg_max; ++arg) {
      EmitFuncProto(os, prefix, inst_name, arg, true) << line_end;
      if (!decl) {
        EmitFuncBody(os, inst_name, arg, true);
      }
    }

    EmitFuncProto(os, prefix, inst_name, -1, true) << line_end;
    if (!decl) {
      EmitFuncBody(os, inst_name, -1, true);
    }
  }
}

} // namespace halo
