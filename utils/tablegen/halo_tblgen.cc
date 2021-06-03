//===- halo_tblgen.cc -------------------------------------------*- C++ -*-===//
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

#include "halo_tblgen.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"
#include "llvm/TableGen/Main.h"
#include "llvm/TableGen/SetTheory.h"

namespace halo {

void EmitSourceFileHeader(const std::string& filename, llvm::raw_ostream& os) {
  const char* license =
      "// Copyright (C) 2019-2020 Alibaba Group Holding Limited.\n"
      "//\n"
      "// Licensed under the Apache License, Version 2.0"
      " (the \"License\");\n"
      "// you may not use this file except in compliance with the"
      " License.\n"
      "// You may obtain a copy of the License at\n"
      "//\n"
      "//   http://www.apache.org/licenses/LICENSE-2.0\n"
      "//\n"
      "// Unless required by applicable law or agreed to in writing,"
      " software\n"
      "// distributed under the License is distributed on an \"AS IS\""
      " BASIS,\n"
      "// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express"
      " or implied.\n"
      "// See the License for the specific language governing permissions"
      " and\n"
      "// limitations under the License.\n"
      "// ==============================================================="
      "==============\n\n";
  const char* tblgen_header =
      "//===- TableGen'erated File ------------------------------------"
      "-*- C++ -*-===//\n"
      "//\n"
      "// Halo IR Code Emitter\n"
      "//\n"
      "// Automatically generated file, do not edit!\n"
      "//\n"
      "// ==============================================================="
      "==============\n\n";
  const char* firstline_prefix = "//===";
  const char* firstline_postfix = "-*- C++ -*-===//\n//\n";
  const int magic = 59;
  auto filename_size = filename.empty() ? 0 : filename.size() + 4;
  auto num_hyphen = magic - filename_size;
  // emit formatted first line.
  os << firstline_prefix;
  if (!filename.empty()) {
    os << "- " << filename << " -";
  }
  while (num_hyphen != 0) {
    os << "-";
    --num_hyphen;
  }
  os << firstline_postfix;
  os << license;
  os << tblgen_header;
}

} // namespace halo

enum ActionType {
  GEN_ATTR_DECL,
  GEN_ATTR_DEF,
  GEN_ATTR_ENUM,
  GEN_CONVERTER_DECL,
  GEN_CONVERTER_DEF,
  GEN_REGISTER_OP,
  GEN_CONVERT_INFO,
  GEN_DATATYPE,
  GEN_DOCUMENT,
  GEN_FUSION,
  GEN_INST_CLASS,
  GEN_INST_INFO,
  GEN_IRBUILDER_DECL,
  GEN_IRBUILDER_DEF,
  GEN_TEST_MODEL,
  GEN_CONFIG_MODEL,
  PRINT_RECORDS,
};

static llvm::cl::opt<ActionType> Action(
    llvm::cl::desc("Action to perform:"),
    llvm::cl::values(
        clEnumValN(GEN_ATTR_DECL, "gen-attr-decl",
                   "Generate attribute class declaration"),
        clEnumValN(GEN_ATTR_DEF, "gen-attr-def",
                   "Generate attribute class definition"),
        clEnumValN(GEN_ATTR_ENUM, "gen-attr-enum",
                   "Generate attribute enum types"),
        clEnumValN(GEN_CONVERT_INFO, "gen-convert-info",
                   "Generate framework op conversion info"),
        clEnumValN(GEN_DATATYPE, "gen-datatype", "Generate enum DataTypeID"),
        clEnumValN(GEN_DOCUMENT, "gen-doc", "Generate IR document"),
        clEnumValN(GEN_FUSION, "gen-fusion", "Generation of fusion code"),
        clEnumValN(GEN_INST_CLASS, "gen-inst-class",
                   "Generation instruction subclass headers"),
        clEnumValN(GEN_INST_INFO, "gen-inst-info",
                   "Generation instruction info defs"),
        clEnumValN(GEN_IRBUILDER_DECL, "gen-irbuilder-decl",
                   "Generation instruction builder headers"),
        clEnumValN(GEN_IRBUILDER_DEF, "gen-irbuilder-def",
                   "Generation instruction builder defs"),
        clEnumValN(GEN_CONVERTER_DECL, "gen-converter-decl",
                   "Generation parser convert function headers"),
        clEnumValN(GEN_CONVERTER_DEF, "gen-converter-def",
                   "Generation parser convert function defs"),
        clEnumValN(GEN_REGISTER_OP, "gen-register-op",
                   "Generation parser supported op register function"),
        clEnumValN(GEN_TEST_MODEL, "gen-test-model", "Generation test model"),
        clEnumValN(GEN_CONFIG_MODEL, "gen-config-model",
                   "Generation config model"),
        clEnumValN(PRINT_RECORDS, "print-records",
                   "Print all records to stdout (default)")));

static bool HaloTableGenMain(llvm::raw_ostream& os,
                             llvm::RecordKeeper& records) { // NOLINT.
  switch (Action) {
    case GEN_ATTR_DECL: {
      halo::EmitAttrDecl(records, os);
      break;
    }
    case GEN_ATTR_DEF: {
      halo::EmitAttrDef(records, os);
      break;
    }
    case GEN_ATTR_ENUM: {
      halo::EmitAttrEnum(records, os);
      break;
    }
    case GEN_CONVERT_INFO: {
      halo::EmitConvertInfo(records, os);
      break;
    }
    case GEN_DATATYPE: {
      halo::EmitDataTypeEnum(records, os);
      break;
    }
    case GEN_DOCUMENT: {
      halo::EmitDoc(records, os);
      break;
    }
    case GEN_FUSION: {
      halo::EmitFusion(records, os);
      break;
    }
    case GEN_INST_CLASS: {
      halo::EmitInstClass(records, os);
      break;
    }
    case GEN_INST_INFO: {
      halo::EmitInstInfo(records, os);
      break;
    }
    case GEN_IRBUILDER_DECL: {
      halo::EmitIRBuilder(records, os, true);
      break;
    }
    case GEN_IRBUILDER_DEF: {
      halo::EmitIRBuilder(records, os, false);
      break;
    }
    case GEN_CONVERTER_DECL: {
      halo::EmitConverterDecl(records, os);
      break;
    }
    case GEN_CONVERTER_DEF: {
      halo::EmitConverterDef(records, os);
      break;
    }
    case GEN_REGISTER_OP: {
      halo::EmitRegisterOp(records, os);
      break;
    }
    case GEN_TEST_MODEL: {
      halo::EmitTestModel(records, os);
      break;
    }
    case GEN_CONFIG_MODEL: {
      halo::EmitConfigModel(records, os);
      break;
    }
    case PRINT_RECORDS: {
      os << records; // No argument, dump all contents
      break;
    }
  }
  return false;
}

int main(int argc, char** argv) {                   // NOLINT.
  llvm::sys::PrintStackTraceOnErrorSignal(argv[0]); // NOLINT.
  llvm::PrettyStackTraceProgram asdf(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv);

  llvm::llvm_shutdown_obj y;
  return TableGenMain(argv[0], &HaloTableGenMain); // NOLINT.
}