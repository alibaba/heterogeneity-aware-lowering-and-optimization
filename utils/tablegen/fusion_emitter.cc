//===- inst_info_emitter.cc -------------------------------------*- C++ -*-===//
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

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

namespace halo {

static void EmitOptions(llvm::raw_ostream* os, llvm::StringRef name) {}

static llvm::StringRef GetDagName(const llvm::DagInit* dag) {
  const llvm::DefInit* def = llvm::cast<llvm::DefInit>(dag->getOperator());
  return def->getDef()->getName();
}

static void EmitMatcher(llvm::raw_ostream* os, llvm::DagInit* pat,
                        llvm::StringRef var_name) {
  int n = pat->arg_size();
  *os << "  if (!ValidateOpSizeAndCode(" << var_name << ", " << n
      << ", OpCode::" << GetDagName(pat).upper() << ")) {return ret;}\n";
  for (int i = 0; i < n; ++i) {
    llvm::Init* arg = pat->getArg(i);
    std::string op_name;
    if (pat->getArgName(i) != nullptr) {
      op_name = pat->getArgNameStr(i).str();
    }
    if (op_name.empty() && llvm::isa<llvm::DagInit>(arg) &&
        llvm::cast<llvm::DagInit>(arg)->getName() != nullptr) {
      op_name = llvm::cast<llvm::DagInit>(arg)->getNameStr().str();
    }
    if (op_name.empty()) {
      op_name = var_name.str() + "_op_" + std::to_string(i);
    }

    *os << "  auto " << op_name << " = " << var_name << "->GetOperand(" << i
        << ");\n";

    if (llvm::isa<llvm::DagInit>(arg)) {
      llvm::DagInit* dag = llvm::cast<llvm::DagInit>(arg);
      std::string op_inst = op_name + "_inst";
      *os << "  if (!IsA<Instruction>(" << op_name << ")) { return ret;}\n";
      *os << "  auto " << op_inst << " = DynCast<Instruction>(" << op_name
          << ");\n";
      EmitMatcher(os, dag, op_inst);
    }
  }
}

static void EmitMatcher(llvm::raw_ostream* os, const llvm::Record* rec) {
  llvm::DagInit* pat = rec->getValueAsDag("pattern_");
  llvm::DagInit* result = rec->getValueAsDag("result_");
  auto rule_name = rec->getName();
  const std::string var_name = "inst";
  *os << "static std::pair<Def, Def> " << rule_name
      << "Matcher(Instruction *inst, IRBuilder *builder) {\n";
  *os << "  std::pair<Def, Def> ret{Def{" << var_name << ", 0}, Def{"
      << var_name << ", 0}};\n";

  EmitMatcher(os, pat, "inst");
  //*os << "if ("
  // Create fusion instr.
  const std::string fused = var_name + "_fused";
  *os << "  builder->SetInsertAfter(" << var_name << ");\n";
  std::string arg_list = "{";
  for (auto i = result->name_begin(), e = result->name_end(); i != e; ++i) {
    arg_list += (*i)->getValue().str() + ", ";
  }
  arg_list += "}";
  *os << "  auto " << fused << " = builder->Create" << GetDagName(result) << "("
      << var_name << "->GetName() + \"_fused\", " << arg_list << "); \n";
  if (auto src = rec->getValueAsString("copy_attrs_from_"); !src.empty()) {
    *os << "  " << fused << "->CopyAttrsFrom(" << src << ");\n";
  }
  *os << "  " << fused << "->GetResultsTypes() = inst->GetResultsTypes();\n";
  *os << "  ret.second = Def(" << fused << ", 0);\n";
  *os << "  return ret; \n}\n";
}

void EmitFusion(const llvm::RecordKeeper& records, llvm::raw_ostream& os) {
  os << "#ifdef HALO_FUSION_MATCHERS\n";
  std::vector<llvm::Record*> fusions =
      records.getAllDerivedDefinitions("Fusion");
  for (auto& rec : fusions) {
    EmitMatcher(&os, rec);
  }
  os << "#endif\n";

  // Emit calls
  os << "\n#ifdef HALO_FUSION_CALLS\n";
  for (auto& rec : fusions) {
    auto rule_name = rec->getName();
    const std::string var_name = "inst";
    os << "  if (ret.first == ret.second && opts_." << rule_name << ") {\n";
    os << "    ret = " << rule_name << "Matcher(inst, &builder);\n";
    os << "  }\n";
  }
  os << "#endif";

  // Emit option member
  os << "\n#ifdef HALO_FUSION_OPTIONS\n";
  for (auto& rec : fusions) {
    auto rule_name = rec->getName();
    os << "    bool " << rule_name << " = false;\n";
  }
  os << "#endif";

  // Emit command line option
  os << "\n#ifdef HALO_FUSION_CMD_OPTIONS_DECL\n";
  for (auto& rec : fusions) {
    auto rule_name = rec->getName();
    os << "static llvm::cl::opt<bool> Fusion" << rule_name << "(\""
       << rec->getValueAsString("option_name_") << "\",\n";
    os << "    llvm::cl::desc(\"" << rec->getValueAsString("option_desc_")
       << "\"), llvm::cl::init(false));\n";
  }
  os << "static Fusion::Options GetFusionOptions() {\n";
  os << "  Fusion::Options opts;";
  for (auto& rec : fusions) {
    auto rule_name = rec->getName();
    os << "    opts." << rule_name << " = Fusion" << rule_name << ";\n";
  }
  os << "  return opts;\n";
  os << "}\n";
  os << "#endif";
}

} // namespace halo