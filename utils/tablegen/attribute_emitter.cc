//===- attribute_emitter.cc -------------------------------------*- C++ -*-===//
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

#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/SetTheory.h"

namespace halo {

// Property names corrresponds to TableGen file.
static const char* RecordTypeName = "ValueType";
static const char* CppType = "cpp_type_";
static const char* IsArray = "is_array_";
static const char* Is2DArray = "is_2d_array_";
static const char* IsPointer = "is_pointer_";
static const char* EnumTypeName = "EnumValueType";
static const char* IsEnum = "is_enum_";

/// This is the main function that emits halo::Attribute subclasses.
/// The emitted code should be included into attribute.h.
void EmitAttrDecl(const llvm::RecordKeeper& records, llvm::raw_ostream& os) {
  std::vector<llvm::Record*> classes =
      records.getAllDerivedDefinitions(RecordTypeName);
  // Emit AttrKind enums.
  os << "  enum class AttrKind {\n";
  for (auto c : classes) {
    os.indent(4);
    os << c->getName().upper() << ",\n";
  }
  os << "    INVALID\n};\n\n";

  // Emit static helper functions for Attribute creation.
  for (auto c : classes) {
    os << "  static std::unique_ptr<Attribute> Create" << c->getName()
       << "(const std::string& name,\n";
    const int indent_size = 43;
    os.indent(indent_size + c->getName().size());
    if (c->getValueAsBit(IsPointer)) {
      os << c->getValueAsString(CppType) << " v);\n";
    } else {
      os << "const " << c->getValueAsString(CppType) << "& v);\n";
    }
  }

  // Emit accessor functions.
  for (auto c : classes) {
    llvm::StringRef cpp_type = c->getValueAsString(CppType);
    llvm::StringRef type_name = c->getName();
    // Emit Getter;
    bool is_pointer = c->getValueAsBit(IsPointer);
    if (is_pointer) {
      os << "\n  " << cpp_type << " GetValueAs" << type_name << "() const {\n";
      os << "    HLCHECK(GetKind() == AttrKind::" << type_name.upper()
         << ");\n";
      os << "    return *static_cast<" << cpp_type
         << "const * >(GetDataImpl());\n";
      os << "  }\n";

      os << "\n  void SetValueAs" << type_name << "(" << cpp_type << " x) {\n";
      os << "    HLCHECK(GetKind() == AttrKind::" << type_name.upper()
         << ");\n";
      os << "    * static_cast<" << cpp_type << "*>(GetDataImpl()) = x; \n ";
      os << "  }\n";
      continue;
    }
    os << "\n  const " << cpp_type << "& GetValueAs" << type_name
       << "() const {\n";
    os << "    HLCHECK(GetKind() == AttrKind::" << type_name.upper() << ");\n";
    os << "    return *static_cast<const " << cpp_type
       << "*>(GetDataImpl());\n";
    os << "  }\n";

    // Emit Setter;
    os << "\n  void SetValueAs" << type_name << "(const " << cpp_type
       << "& x) {\n";

    os << "    HLCHECK(GetKind() == AttrKind::" << type_name.upper() << ");\n";
    os << "    *static_cast<" << cpp_type << "*>(GetDataImpl()) = x;\n";
    os << "  }\n";
  }
}

/// Emits the declaration of subclasses of Attribute class.
/// The emitted code should be included into attribute.cpp.
void EmitAttrDef(const llvm::RecordKeeper& records, llvm::raw_ostream& os) {
  std::vector<llvm::Record*> classes =
      records.getAllDerivedDefinitions(RecordTypeName);
  for (auto c : classes) {
    llvm::StringRef name = c->getName();
    llvm::StringRef cpp_type = c->getValueAsString("cpp_type_");
    bool is_pointer = c->getValueAsBit(IsPointer);

    os << "\nclass Attribute" << name << " final : public Attribute {\n";
    os << " public:\n";
    os << "  Attribute" << name << "(const std::string& name, ";
    if (is_pointer) {
      os << cpp_type << " x)\n";
    } else {
      os << "const " << cpp_type << "& x)\n";
    }
    os << "      : Attribute(name), value_(x) {}\n";
    os << "  void* GetDataImpl() override { return &value_; }\n";
    os << "  const void* GetDataImpl() const override { return &value_; "
          "}\n";

    os << "  std::unique_ptr<Attribute> Clone() const override {\n";
    os << "    return std::move(Attribute::Create" << name
       << "(GetName(), value_));\n";
    os << "  }\n";
    os << "  void Print(std::ostream& os) const override {\n";
    os << "    os << \"<\" << GetName() << \": \";\n";
    if (c->getValueAsBit(IsArray)) {
      os << "    os << \"[\";\n";
      os << "    size_t num_of_elements = value_.size();\n";
      os << "    if (num_of_elements > 0) {\n";
      if (c->getValueAsBit(IsEnum)) {
        os << "      os << \"enum_data_type \" << (int)value_.at(0);\n";
        os << "      for (size_t i = 1; i < num_of_elements; ++i) {\n";
        os << "        os << \", \" << (int)value_.at(i);\n";
        os << "      }\n";
      } else {
        os << "      os << value_.at(0);\n";
        os << "      for (size_t i = 1; i < num_of_elements; ++i) {\n";
        os << "        os << \", \" << value_.at(i);\n";
        os << "      }\n";
      }

      os << "    }\n";
      os << "    os << \"]>\";\n";
    } else if (c->getValueAsBit(Is2DArray)) {
      os << "    os << \"[\";\n";
      os << "    size_t num_of_lists = value_.size();\n";
      os << "    if (num_of_lists > 0) {\n";
      os << "      for (size_t i = 0; i < num_of_lists; ++i) {\n";
      os << "        size_t num_of_elements = value_.at(i).size();\n";
      os << "        if (num_of_elements > 0) {\n";
      os << "          os << value_.at(i).at(0);\n";
      os << "          for (size_t j = 1; j < num_of_elements; ++j) {\n";
      os << "            os << \", \" << value_.at(i).at(j);\n";
      os << "          }\n";
      os << "        }\n";
      os << "      }\n";
      os << "    }\n";
      os << "    os << \"]>\";\n";
    } else {
      if (c->isSubClassOf(EnumTypeName)) {
        os << "    os << static_cast<int>(value_);\n";
      } else {
        os << "    os << value_ << \">\";\n";
      }
    }

    os << "  }\n";
    os << "  AttrKind GetKind() const override { return AttrKind::"
       << name.upper() << "; }\n";
    os << "\n private:\n";
    os << "  " << cpp_type << " value_;\n";
    os << "};\n";

    // Emit static creator.
    os << "\nstd::unique_ptr<Attribute> Attribute::Create" << name << "(\n";
    os << "    const std::string& name, ";
    if (c->getValueAsBit(IsPointer)) {
      os << c->getValueAsString(CppType) << " v) {\n";
    } else {
      os << "const " << c->getValueAsString(CppType) << "& v) {\n";
    }
    os << "  return std::make_unique<Attribute" << name << ">(name, v);\n";
    os << "}\n";
  }
}

} // namespace halo