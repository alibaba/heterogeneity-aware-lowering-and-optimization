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

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

// Generate ExtOpCode from Extension record name, e.g.,
// TF_Broadcast ==> BROADCAST
static std::string Normalize(llvm::StringRef record_name) {
  size_t pre_len = record_name.find_first_of('_');
  if (pre_len != llvm::StringRef::npos) {
    return record_name.drop_front(pre_len + 1).upper();
  }
  llvm_unreachable("FRAMEWORK_OPNAME format is expected.");
  return "";
}

namespace halo {

/// Emit IR visitors like:
///   virtual void RunOnInstruction(AddInst&) = 0;
static void EmitRunOnInstruction(const std::vector<llvm::Record*>& insts,
                                 llvm::raw_ostream& os) {
  const char* macro = "GET_RUN_ON_INSTRUCTION_DECL";
  os << "#ifdef " << macro << "\n";
  for (auto& inst : insts) {
    os.indent(2);
    os << "virtual void RunOnInstruction(" << inst->getName()
       << "Inst*) { HLCHECK(0); };\n";
  }
  os << "#endif //" << macro << "\n\n";
}

/// Emit casting switch body like
///    case OpCode::ADD: {
///      RunOnInstruction(*static_cast<const AddInst*>(&inst));
///      break;
///   }
namespace {
enum class Option {
  WITH_RETURN,
  WITHOUT_RETURN,
  TAKE_EXTRA_PARAM,
};
} // namespace

static void EmitCastingSwitch(const std::vector<llvm::Record*>& insts,
                              llvm::raw_ostream& os, const Option& opt) {
  const char* macro = "GET_INST_DOWNCAST_SWITCH_WITH_RETURN";
  switch (opt) {
    case Option::WITH_RETURN:
      break;
    case Option::WITHOUT_RETURN:
      macro = "GET_INST_DOWNCAST_SWITCH";
      break;
    case Option::TAKE_EXTRA_PARAM:
      macro = "GET_INST_DOWNCAST_SWITCH_TAKE_EXTRA_PARAM";
      break;
    default:
      break;
  }

  os << "#ifdef " << macro << "\n";
  for (auto& inst : insts) {
    os << "    case OpCode::" << inst->getName().upper() << ": {\n";
    os << "      ";
    if (opt == Option::WITH_RETURN) {
      os << "ret = ";
    }

    os << "RunOnInstruction(static_cast<" << inst->getName() << "Inst*>(inst)";
    if (opt == Option::TAKE_EXTRA_PARAM) {
      os << ", &node_infos_";
    }
    os << ");\n";
    os << "      break;\n";
    os << "    }\n";
  }
  os << "#endif // " << macro << "\n\n";
}

/// Emit enum values for OpCode.
/// Each Inst def record generates a value.
void EmitInstInfo(const llvm::RecordKeeper& records, llvm::raw_ostream& os) {
  std::vector<llvm::Record*> insts = records.getAllDerivedDefinitions("Inst");
  std::sort(insts.begin(), insts.end(),
            [](llvm::Record* lhs, llvm::Record* rhs) {
              auto op1 = lhs->getName();
              auto op2 = rhs->getName();
              return (op1.compare(op2) <= 0);
            });

  os << "// Enum values for OpCode\n";
  os << "#ifdef GET_INST_INFO_OPCODE_ENUM\n";
  for (auto inst : insts) {
    os << inst->getName().upper() << ",\n";
  }

  os << "#endif // GET_INST_INFO_OPCODE_ENUM\n\n";

  os << "// print name for OpCode\n";
  os << "#ifdef GET_INST_INFO_OPCODE_PRINT\n";
  for (auto inst : insts) {
    auto op = inst->getName();
    os << "case OpCode::" << op.upper() << ": os << ";
    os << "\"" << op.lower() << "\"; break;\n";
  }
  os << "#endif // GET_INST_INFO_OPCODE_PRINT\n\n";

  EmitCastingSwitch(insts, os, Option::WITH_RETURN);
  EmitCastingSwitch(insts, os, Option::WITHOUT_RETURN);
  EmitCastingSwitch(insts, os, Option::TAKE_EXTRA_PARAM);

  EmitRunOnInstruction(insts, os);
}

void EmitConvertInfo(const llvm::RecordKeeper& records, llvm::raw_ostream& os) {
  std::vector<llvm::Record*> extensions =
      records.getAllDerivedDefinitions("OpExtension");
  std::vector<std::string> frameworks = {"TF", "CAFFE", "ONNX", "MXNET",
                                         "TFLITE"};
  std::string framework = "CUSTOM";
  if (!extensions.empty()) {
    for (const auto& fw : frameworks) {
      if (extensions[0]->isSubClassOf(fw + "Extension")) {
        framework = fw;
        break;
      }
    }
  }
  std::string ifdef = "GET_" + framework + "_OPCODE_ENUM";
  os << " // Enums for ExtOpCode\n";
  os << "#ifdef " << ifdef << "\n";
  for (auto r : extensions) {
    os << Normalize(r->getName()) << ", \n";
  }
  os << "#endif // " << ifdef << "\n\n";

  ifdef = "GET_" + framework + "_NAME_OPCODE_MAPPING";
  os << " // Name to ExtOpCode mapping\n";
  os << "#ifdef " << ifdef << "\n";
  for (auto r : extensions) {
    auto name = r->getValueAsString("extern_op_");
    os << "{\"" << name << "\", ";
    os << framework << "ExtOpCode::" << Normalize(r->getName()) << "}, \n";
  }
  os << "#endif // " << ifdef << "\n\n";
}

} // namespace halo