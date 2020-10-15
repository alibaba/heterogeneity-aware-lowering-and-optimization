//===- enum_emitter.cc ------------------------------------------*- C++ -*-===//
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

#include <string>

#include "inst.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

namespace halo {

/// Emit one enum declaration
void EmitOneEnum(const llvm::Record* record, llvm::raw_ostream& os) {
  std::string name = record->getValueAsString("cpp_type_").str();
  // special handling of DataType
  if (name == "DataType") {
    return;
  }
  std::vector<llvm::StringRef> enums =
      record->getValueAsListOfStrings("valid_enums_");
  os << "enum class " << name << " {\n";
  for (const auto& e : enums) {
    os.indent(2);
    os << e;
    os << ",\n";
  }
  // always emit an INVALID value at last.
  os.indent(2);
  os << "INVALID,\n";
  os << "};\n\n";
}

/// Emit attribute enum types from attribute_types.td
void EmitAttrEnum(const llvm::RecordKeeper& records, llvm::raw_ostream& os) {
  std::vector<llvm::Record*> enums =
      records.getAllDerivedDefinitions("EnumValueType");
  for (auto ib = enums.begin(), ie = enums.end(); ib != ie; ++ib) {
    EmitOneEnum(*ib, os);
  }
}

/// Emit DataTypeID enum from arg_types.td
void EmitDataTypeEnum(const llvm::RecordKeeper& records,
                      llvm::raw_ostream& os) {
  std::vector<llvm::Record*> types =
      records.getAllDerivedDefinitions("ValueType");
  std::vector<tablegen::Type> datatypes;
  datatypes.reserve(types.size());
  for (auto r : types) {
    datatypes.emplace_back(r);
  }
  std::sort(datatypes.begin(), datatypes.end(),
            [](tablegen::Type& lhs, tablegen::Type& rhs) {
              if (lhs.IsString != rhs.IsString) {
                return lhs.IsString;
              }
              if (lhs.IsBool != rhs.IsBool) {
                return lhs.IsBool;
              }
              if (lhs.IsFloat != rhs.IsFloat) {
                return lhs.IsFloat;
              }
              if (lhs.IsUnsigned != rhs.IsUnsigned) {
                return !lhs.IsUnsigned;
              }
              if (lhs.IsQuantized != rhs.IsQuantized) {
                return !lhs.IsQuantized;
              }
              return lhs.Width < rhs.Width;
            });
  std::vector<std::string> enum_strs;
  enum_strs.reserve(datatypes.size());
  for (auto vt : datatypes) {
    enum_strs.push_back("");
    auto& name = enum_strs.back();
    if (vt.IsBool) {
      name = "BOOL";
    } else if (vt.IsString) {
      name = "STRING";
    } else {
      if (vt.IsFloat) {
        name = "FLOAT";
      } else if (vt.IsInt) {
        if (vt.IsQuantized) {
          name += "Q";
        }
        if (vt.IsUnsigned) {
          name += "U";
        }
        name += "INT";
      }
      name += std::to_string(vt.Width);
    }
  }
  os << "#ifdef GET_DATATYPE_ENUM_VALUE\n";
  std::for_each(enum_strs.begin(), enum_strs.end(),
                [&](const std::string& s) { os << s << ",\n"; });
  os << "#endif // GET_DATATYPE_ENUM_VALUE\n";

  os << "\n";
  os << "#ifdef GET_DATATYPE_ENUM_STRING\n";
  for (const auto& s : enum_strs) {
    os << "case DataType::" << s << ": ";
    os << "s = "
       << "\"" << s << "\"";
    os << "; break;\n";
  }
  os << "#endif // GET_DATATYPE_ENUM_STRING\n";

  os << "\n";
  os << "#ifdef GET_DATATYPE_ENUM_FROM_STRING\n";
  for (const auto& s : enum_strs) {
    os << "if (s == \"" << s << "\") {\n";
    os << "  return DataType::" << s << ";\n";
    os << "}\n";
  }
  os << "#endif // GET_DATATYPE_ENUM_FROM_STRING\n";
}

} // end namespace halo