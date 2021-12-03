//===- converter_emitter.cc -------------------------------------*- C++ -*-===//
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

#include <iostream>
#include <sstream>
#include <unordered_set>

#include "inst.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Record.h"

namespace halo {

// Property names corrresponds to TableGen file.
static const char* RecordTypeName = "OpMapping";
static const char* ExternOpName = "extern_op_";
static const char* SNInst = "sn_inst_";
static const char* CustomCode = "custom_code_";
static const char* ParamName = "param_name_";
static const char* OptionalAttrs = "optional_attrs_";
static const char* NumOutputs = "num_outputs_";

using AttrValue = std::pair<std::string, std::string>;

namespace tablegen {

class Converter {
 public:
  explicit Converter(const llvm::RecordKeeper& records);
  void EmitRegisterOp(llvm::raw_ostream& os);
  void EmitConverterDecl(llvm::raw_ostream& os);
  void EmitConverterDef(llvm::raw_ostream& os);
  enum class Framework {
    FRAMEWORK_TF,
    FRAMEWORK_ONNX,
    FRAMEWORK_TFLITE,
    FRAMEWORK_CAFFE,
    FRAMEWORK_MXNET,
  };

 private:
  void EmitExtensionInstDef(llvm::Record* record, llvm::raw_ostream& os);
  void EmitExtensionInstDefForCaffe(llvm::Record* record,
                                    llvm::raw_ostream& os);
  void EmitHaloInstDef(llvm::Record* record, llvm::raw_ostream& os);
  void EmitConverterDefForCaffe(llvm::raw_ostream& os);
  void EmitHaloInstDefForCaffe(llvm::Record* record, llvm::raw_ostream& os);
  static void ProcessExtensionAttributes(
      const std::vector<llvm::Record*>& extension_attrs,
      const std::string& framework, const std::string& param_name,
      llvm::raw_ostream& os);
  static void ProcessExtensionAttributesImpl(
      const std::vector<llvm::Record*>& extension_attrs,
      const std::string& framework, const std::string& param_name,
      llvm::raw_ostream& os, bool need_new_attr);
  static void ProcessExtensionAttributesForCaffe(
      const std::vector<llvm::Record*>& extension_attrs,
      const std::string& param_name,
      const std::unordered_set<std::string>& optional_attrs,
      llvm::raw_ostream& os);
  static void ProcessAttributes(
      llvm::Record* inst,
      const std::unordered_map<std::string, llvm::Record*>& need_mapping,
      llvm::raw_ostream& os, bool* need_new_attr);
  static void ProcessAttributesForCaffe(
      llvm::Record* inst,
      const std::unordered_map<std::string, llvm::Record*>& need_mapping,
      const std::string& param_name,
      const std::unordered_set<std::string>& optional_attrs,
      llvm::raw_ostream& os);

  const llvm::RecordKeeper& records_;
  Framework framework_kind_;
  std::string framework_name_;
  std::string framework_namespace_;
  std::string pb_node_ = "INVALID";
};

Converter::Converter(const llvm::RecordKeeper& records) : records_(records) {
  std::vector<llvm::Record*> convert_records =
      records.getAllDerivedDefinitions(RecordTypeName);
  auto one = convert_records[0];
  if (one->getName().startswith("TFLITE")) {
    framework_kind_ = Framework::FRAMEWORK_TFLITE;
    framework_name_ = "TFLITE";
    framework_namespace_ = "tflite";
    pb_node_ = "Operator";
  } else if (one->getName().startswith("TF")) {
    framework_kind_ = Framework::FRAMEWORK_TF;
    framework_name_ = "TF";
    framework_namespace_ = "tensorflow";
    pb_node_ = "NodeDef";
  } else if (one->getName().startswith("ONNX")) {
    framework_kind_ = Framework::FRAMEWORK_ONNX;
    framework_name_ = "ONNX";
    framework_namespace_ = "onnx";
    pb_node_ = "NodeProto";
  } else if (one->getName().startswith("CAFFE")) {
    framework_kind_ = Framework::FRAMEWORK_CAFFE;
    framework_name_ = "CAFFE";
    framework_namespace_ = "caffe";
    pb_node_ = "LayerParameter";
  } else if (one->getName().startswith("MXNET")) {
    framework_kind_ = Framework::FRAMEWORK_MXNET;
    framework_name_ = "MXNET";
    framework_namespace_ = "mxnet";
  } else {
    llvm_unreachable("unsupported framework.");
  }
}

/// Emits the register op functions
void Converter::EmitRegisterOp(llvm::raw_ostream& os) {
  std::vector<llvm::Record*> classes =
      records_.getAllDerivedDefinitions(RecordTypeName);
  const std::string parser = framework_name_ + "Parser";
  for (const auto& c : classes) {
    llvm::StringRef extern_op_name = c->getValueAsString(ExternOpName);
    os << "func_lists_.emplace(\"" << extern_op_name << "\", std::bind(&"
       << parser << "::Convert" << extern_op_name
       << "Node, this, std::placeholders::_1, std::placeholders::_2";
    if (framework_kind_ == Framework::FRAMEWORK_CAFFE) {
      os << ", std::placeholders::_3));\n";
    } else {
      os << "));\n";
    }
  }
}

/// Emit convert function declare
void Converter::EmitConverterDecl(llvm::raw_ostream& os) {
  std::vector<llvm::Record*> classes =
      records_.getAllDerivedDefinitions(RecordTypeName);
  for (const auto& c : classes) {
    llvm::StringRef extern_op_name = c->getValueAsString(ExternOpName);
    os << "Status Convert" << extern_op_name
       << "Node(IRBuilder* ir_builder, const " << framework_namespace_
       << "::" << pb_node_ << "& node_def";
    if (framework_kind_ == Framework::FRAMEWORK_CAFFE) {
      os << ", const " << framework_namespace_ << "::" << pb_node_
         << "& layer_param_weight);\n";
    } else {
      os << ");\n";
    }
  }
}

void Converter::EmitConverterDef(llvm::raw_ostream& os) {
  if (framework_kind_ == Framework::FRAMEWORK_CAFFE) {
    EmitConverterDefForCaffe(os);
    return;
  }
  std::vector<llvm::Record*> classes =
      records_.getAllDerivedDefinitions(RecordTypeName);
  for (auto c : classes) {
    if (c->isSubClassOf("TFExtension") || c->isSubClassOf("ONNXExtension") ||
        c->isSubClassOf("TFLITEExtension")) {
      EmitExtensionInstDef(c, os);
    } else if (c->isSubClassOf("MXNETExtension")) {
      llvm_unreachable("Unimplemented.");
    } else {
      EmitHaloInstDef(c, os);
    }
  }
}

void Converter::EmitConverterDefForCaffe(llvm::raw_ostream& os) {
  std::vector<llvm::Record*> classes =
      records_.getAllDerivedDefinitions(RecordTypeName);
  for (auto c : classes) {
    if (c->isSubClassOf("CAFFEExtension")) {
      EmitExtensionInstDefForCaffe(c, os);
    } else {
      EmitHaloInstDefForCaffe(c, os);
    }
  }
}

void Converter::EmitHaloInstDefForCaffe(llvm::Record* record,
                                        llvm::raw_ostream& os) {
  llvm::Record* sn_inst = record->getValueAsDef(SNInst);
  llvm::StringRef sn_inst_name = sn_inst->getName();
  llvm::StringRef extern_op_name = record->getValueAsString(ExternOpName);
  llvm::StringRef param_name = record->getValueAsString(ParamName);
  std::vector<llvm::StringRef> optional_attrs =
      record->getValueAsListOfStrings(OptionalAttrs);
  std::unordered_set<std::string> attrs_set;
  for (auto ref : optional_attrs) {
    attrs_set.insert(ref.str());
  }

  const std::string parser = framework_name_ + "Parser";
  os << "Status " << parser << "::Convert" << extern_op_name
     << "Node(IRBuilder* ir_builder, const " << framework_namespace_
     << "::" << pb_node_ << "& node_def, const " << framework_namespace_
     << "::" << pb_node_ << "& layer_param_weight) {\n";

  std::unordered_map<std::string, llvm::Record*> need_mapping;
  std::vector<llvm::Record*> attr_mapping =
      record->getValueAsListOfDefs("attr_mapping_");
  for (const auto& it : attr_mapping) {
    need_mapping.emplace(it->getValueAsString("sn_attr_").str(), it);
  }

  os << "  std::vector<Def> operands = "
        "GetInputOperands(node_def, layer_param_weight);\n";
  os << "  auto inst = ir_builder->Create" << sn_inst_name
     << "(node_def.name(), operands);\n";
  ProcessAttributesForCaffe(sn_inst, need_mapping, param_name.str(), attrs_set,
                            os);
  os << "  InsertIDToInstMap(node_def, inst);\n";
  os << "  return Status::SUCCESS;\n}\n\n";
}

void Converter::EmitHaloInstDef(llvm::Record* record, llvm::raw_ostream& os) {
  llvm::Record* sn_inst = record->getValueAsDef(SNInst);
  llvm::StringRef sn_inst_name = sn_inst->getName();
  llvm::StringRef extern_op_name = record->getValueAsString(ExternOpName);
  const std::string parser = framework_name_ + "Parser";
  os << "Status " << parser << "::Convert" << extern_op_name
     << "Node(IRBuilder* ir_builder, const " << framework_namespace_
     << "::" << pb_node_ << "& node_def) {\n";

  std::unordered_map<std::string, llvm::Record*> need_mapping;
  std::vector<llvm::Record*> attr_mapping =
      record->getValueAsListOfDefs("attr_mapping_");
  for (const auto& it : attr_mapping) {
    need_mapping.emplace(it->getValueAsString("sn_attr_").str(), it);
  }

  std::vector<llvm::Record*> attrs = sn_inst->getValueAsListOfDefs("attrs_");
  if (!attrs.empty() && framework_name_ != "TFLITE") {
    os << "  " << framework_name_ << "Attrs attrs(node_def);\n";
  }
  os << "  std::vector<Def> operands = GetInputOperands(node_def);\n";
  if (framework_name_ != "TFLITE") {
    os << "  auto name = node_def.name();\n";
  }
  if (framework_name_ == "ONNX") {
    os << "  name = (name.empty() && node_def.output_size() > 0) ? "
          "node_def.output(0) : name;\n";
  }
  os << "  auto inst = ir_builder->Create" << sn_inst_name;
  if (framework_name_ == "TFLITE") {
    os << "(\"\", operands);\n"; // Instruction constructor auto set inst name
  } else {
    os << "(name, operands);\n";
  }

  bool need_new_attr = true;
  ProcessAttributes(sn_inst, need_mapping, os, &need_new_attr);
  std::vector<llvm::Record*> extension_attrs =
      record->getValueAsListOfDefs("extension_attr_");
  llvm::StringRef param_name = record->getValueAsString(ParamName);
  ProcessExtensionAttributesImpl(extension_attrs, framework_name_,
                                 param_name.str(), os, need_new_attr);

  if (auto code = record->getValueAsString(CustomCode); !code.empty()) {
    os << code;
  }

  os << "  InsertIDToInstMap(node_def, inst);\n";
  os << "  return Status::SUCCESS;\n}\n\n";
}

void Converter::EmitExtensionInstDef(llvm::Record* record,
                                     llvm::raw_ostream& os) {
  llvm::Record* sn_inst = record->getValueAsDef(SNInst);
  llvm::StringRef sn_inst_name = sn_inst->getName();
  llvm::StringRef extern_op_name = record->getValueAsString(ExternOpName);
  llvm::StringRef param_name = record->getValueAsString(ParamName);
  int num_outputs = record->getValueAsInt(NumOutputs);

  os << "Status " << framework_name_ << "Parser::Convert" << extern_op_name
     << "Node(IRBuilder* ir_builder, const " << framework_namespace_
     << "::" << pb_node_ << "& node_def) {\n";
  std::vector<llvm::Record*> extension_attrs =
      record->getValueAsListOfDefs("extension_attr_");

  os << "  std::vector<Def> operands = GetInputOperands(node_def);\n";
  if (framework_name_ != "TFLITE") {
    os << "  auto name = node_def.name();\n";
  }
  if (framework_name_ == "ONNX") {
    os << "  name = (name.empty() && node_def.output_size() > 0) ? "
          "node_def.output(0) : name;\n";
  }
  os << "  auto inst = ir_builder->Create" << framework_name_ << sn_inst_name;
  if (framework_name_ == "TFLITE") {
    os << "(\"\""; // Instruction constructor auto set inst name
  } else {
    os << "(name";
  }
  os << ", operands, " << num_outputs << ", \"" << extern_op_name << "\");\n";

  ProcessExtensionAttributes(extension_attrs, framework_name_, param_name.str(),
                             os);

  if (auto code = record->getValueAsString(CustomCode); !code.empty()) {
    os << code;
  }

  os << "  InsertIDToInstMap(node_def, inst);\n";
  os << "  return Status::SUCCESS;\n}\n\n";
}

void Converter::EmitExtensionInstDefForCaffe(llvm::Record* record,
                                             llvm::raw_ostream& os) {
  llvm::Record* sn_inst = record->getValueAsDef(SNInst);
  llvm::StringRef sn_inst_name = sn_inst->getName();
  llvm::StringRef extern_op_name = record->getValueAsString(ExternOpName);
  llvm::StringRef param_name = record->getValueAsString(ParamName);
  std::vector<llvm::StringRef> optional_attrs =
      record->getValueAsListOfStrings(OptionalAttrs);
  std::unordered_set<std::string> attrs_set;
  for (auto ref : optional_attrs) {
    attrs_set.insert(ref.str());
  }
  os << "Status " << framework_name_ << "Parser::Convert" << extern_op_name
     << "Node(IRBuilder* ir_builder, const " << framework_namespace_
     << "::" << pb_node_ << "& node_def, const " << framework_namespace_
     << "::" << pb_node_ << "& layer_param_weight) {\n";
  std::vector<llvm::Record*> extension_attrs =
      record->getValueAsListOfDefs("extension_attr_");

  os << "  std::vector<Def> operands = GetInputOperands(node_def, "
        "layer_param_weight);\n";
  os << "  auto inst = ir_builder->Create" << framework_name_ << sn_inst_name
     << "(node_def.name(), operands, 1, \"" << extern_op_name << "\");\n";

  ProcessExtensionAttributesForCaffe(extension_attrs, param_name.str(),
                                     attrs_set, os);
  os << "  InsertIDToInstMap(node_def, inst);\n";
  os << "  return Status::SUCCESS;\n}\n\n";
}

void Converter::ProcessExtensionAttributesForCaffe(
    const std::vector<llvm::Record*>& extension_attrs,
    const std::string& param_name,
    const std::unordered_set<std::string>& optional_attrs,
    llvm::raw_ostream& os) {
  if (extension_attrs.empty()) {
    return;
  }
  constexpr int indent_num = 2;
  os << "  const auto& attrs = node_def." << param_name << "();\n";
  for (const auto& attr : extension_attrs) {
    llvm::Record* value_type = attr->getValueAsDef("value_type_");
    const std::string cpp_type =
        value_type->getValueAsString("cpp_type_").str();
    llvm::StringRef extern_attr_name =
        attr->getValueAsString("extern_attr_name_");
    std::string default_value = attr->getValueAsString("default_value_").str();
    if (optional_attrs.count(extern_attr_name.str()) != 0) {
      os << "  if(attrs.has_" << extern_attr_name << "()) {\n";
      os.indent(2);
    }
    os.indent(indent_num);
    os << cpp_type << " " << extern_attr_name;
    if (value_type->getValueAsBit("is_array_")) {
      os << ";\n";
      // TODO (unknown): handle BlobShape
      if (extern_attr_name == "shape") {
        os.indent(indent_num + 2);
        os << "CAFFEAttrs attr_shape(attrs." << extern_attr_name << "());\n";
        os.indent(indent_num + 2);
        os << extern_attr_name << " = attr_shape.GetShape();\n";
      } else {
        os.indent(indent_num + 2);
        os << "for (int i = 0; i < attrs." << extern_attr_name
           << "().size(); ++i) {\n";
        os.indent(indent_num + 2);
        os << extern_attr_name << ".push_back(attrs." << extern_attr_name
           << "(i))"
           << ";\n";
        os.indent(indent_num);
        os << "}\n";
      }
    } else {
      os << " = "
         << "attrs." << extern_attr_name << "();\n";
    }
    os.indent(indent_num);
    os << "inst->AddOneAttribute(Attribute::Create";
    os << value_type->getName() << "(\"" << extern_attr_name << "\", "
       << extern_attr_name << "));\n";
    if (optional_attrs.count(extern_attr_name.str()) != 0) {
      os << "  }\n";
    }
  }
}

void Converter::ProcessExtensionAttributes(
    const std::vector<llvm::Record*>& extension_attrs,
    const std::string& framework_name, const std::string& param_name,
    llvm::raw_ostream& os) {
  constexpr int indent_num = 2;
  bool flag = true;
  if (!extension_attrs.empty() && framework_name != "CAFFE" &&
      framework_name != "TFLITE") {
    os << "  " << framework_name << "Attrs attrs(node_def);\n";
  } else if (!extension_attrs.empty() && framework_name == "TFLITE") {
    os << "  const auto& attrs = *(node_def." << param_name << "());\n";
    flag = false;
  }
  ProcessExtensionAttributesImpl(extension_attrs, framework_name, param_name,
                                 os, flag);
}

void Converter::ProcessExtensionAttributesImpl(
    const std::vector<llvm::Record*>& extension_attrs,
    const std::string& framework_name, const std::string& param_name,
    llvm::raw_ostream& os, bool need_new_attr) {
  constexpr int indent_num = 2;
  if (!extension_attrs.empty() && framework_name == "TFLITE" && need_new_attr) {
    os << "  const auto& attrs = *(node_def." << param_name << "());\n";
  }
  for (const auto& attr : extension_attrs) {
    llvm::Record* value_type = attr->getValueAsDef("value_type_");
    const std::string cpp_type =
        value_type->getValueAsString("cpp_type_").str();
    llvm::StringRef extern_attr_name =
        attr->getValueAsString("extern_attr_name_");
    std::string default_value = attr->getValueAsString("default_value_").str();
    if (value_type->isSubClassOf("EnumValueType")) {
      std::ostringstream ss;
      ss << cpp_type << "::" << default_value;
      default_value = ss.str();
    }
    os.indent(indent_num);
    os << cpp_type << " " << extern_attr_name << " = " << default_value
       << ";\n";
    os.indent(indent_num);
    if (framework_name == "TFLITE") {
      if (value_type->isSubClassOf("EnumValueType") ||
          value_type->getValueAsBit("is_array_")) {
        os << extern_attr_name << " = process_" << extern_attr_name << "(attrs."
           << extern_attr_name << "());\n";
      } else {
        os << extern_attr_name << " = "
           << "attrs." << extern_attr_name << "();\n";
      }
    } else {
      os << "attrs.Process<" << cpp_type << ">(\"" << extern_attr_name
         << "\", &" << extern_attr_name << ");\n";
    }
    os.indent(indent_num);
    os << "inst->AddOneAttribute(Attribute::Create";
    os << value_type->getName() << "(\"" << extern_attr_name << "\", "
       << extern_attr_name << "));\n";
  }
}

static void SetAttribute(const std::string& attr_name, const std::string& value,
                         bool spatial_only, int element_index, int indent,
                         llvm::raw_ostream& os) {
  if (spatial_only) {
    // TODO(unknown): expand dim according to data format
    os.indent(indent);
    os << value << ".insert(" << value << ".begin(), 2, 1);\n";
  }
  std::string attr_value = value;
  if (element_index != -1) {
    attr_value += ".at(" + std::to_string(element_index) + ")";
  }
  os.indent(indent);
  os << "inst->Set" << tablegen::Attr::SetAccessName(attr_name) << "("
     << attr_value << ");\n";
}

void Converter::ProcessAttributes(
    llvm::Record* inst,
    const std::unordered_map<std::string, llvm::Record*>& need_mapping,
    llvm::raw_ostream& os, bool* need_new_attr) {
  constexpr int indent_num = 2;
  std::vector<llvm::Record*> attrs = inst->getValueAsListOfDefs("attrs_");
  for (const auto& attr : attrs) {
    const std::string sn_attr_name = attr->getValueAsString("attr_name_").str();
    llvm::Record* type = attr->getValueAsDef("type_");
    std::string cpp_type = type->getValueAsString("cpp_type_").str();

    if (const auto& it = need_mapping.find(sn_attr_name);
        it != need_mapping.end()) {
      llvm::Record* attr_mapping_record = it->second;
      const std::string extern_attr_name =
          attr_mapping_record->getValueAsString("extern_attr_name_").str();
      std::string default_value =
          attr_mapping_record->getValueAsString("attr_value_").str();
      bool spatial_only = attr_mapping_record->getValueAsBit("expand_dims_");
      int index = attr_mapping_record->getValueAsInt("index_");
      if (index != -1) {
        std::ostringstream ss;
        ss << "std::vector<" << cpp_type << ">";
        cpp_type = ss.str();
      }
      if (type->isSubClassOf("EnumValueType")) {
        std::ostringstream ss;
        ss << cpp_type << "::" << default_value;
        default_value = ss.str();
      }
      if (!extern_attr_name.empty()) {
        // need mapping, whether sn_attr_name and extern_attr_name is the same
        // or not
        os.indent(indent_num);
        os << cpp_type << " " << sn_attr_name << " = " << default_value
           << ";\n";
        os.indent(indent_num);
        os << "attrs.Process<" << cpp_type << ">(\"" << extern_attr_name
           << "\", &" << sn_attr_name << ");\n";
        SetAttribute(sn_attr_name, sn_attr_name, spatial_only, index,
                     indent_num, os);
        os.indent(indent_num);
        os << "\n";
        *need_new_attr = false;
      } else {
        // no need to mapping attribute, such as CMP inst
        os.indent(indent_num);
        os << "inst->Set" << tablegen::Attr::SetAccessName(sn_attr_name) << "("
           << default_value << ");\n";
      }
    }
  }
}

void Converter::ProcessAttributesForCaffe(
    llvm::Record* inst,
    const std::unordered_map<std::string, llvm::Record*>& need_mapping,
    const std::string& param_name,
    const std::unordered_set<std::string>& optional_attrs,
    llvm::raw_ostream& os) {
  constexpr int indent_num = 2;
  std::vector<llvm::Record*> attrs = inst->getValueAsListOfDefs("attrs_");
  if (!param_name.empty()) {
    os << "  const auto& attrs = node_def." << param_name << "();\n";
  }

  for (const auto& attr : attrs) {
    const std::string sn_attr_name = attr->getValueAsString("attr_name_").str();
    llvm::Record* type = attr->getValueAsDef("type_");
    std::string cpp_type = type->getValueAsString("cpp_type_").str();

    if (const auto& it = need_mapping.find(sn_attr_name);
        it != need_mapping.end()) {
      llvm::Record* attr_mapping_record = it->second;
      const std::string extern_attr_name =
          attr_mapping_record->getValueAsString("extern_attr_name_").str();
      std::string default_value =
          attr_mapping_record->getValueAsString("attr_value_").str();
      bool spatial_only = attr_mapping_record->getValueAsBit("expand_dims_");
      int index = attr_mapping_record->getValueAsInt("index_");
      if (index != -1) {
        std::ostringstream ss;
        ss << "std::vector<" << cpp_type << ">";
        cpp_type = ss.str();
      }
      if (type->isSubClassOf("EnumValueType")) {
        std::ostringstream ss;
        ss << cpp_type << "::" << default_value;
        default_value = ss.str();
      }
      if (!extern_attr_name.empty()) {
        // need mapping, whether sn_attr_name and extern_attr_name is the same
        // or not
        os.indent(indent_num);
        if (optional_attrs.count(extern_attr_name) != 0) {
          os << "if(attrs.has_" << extern_attr_name << "()) {\n";
          os.indent(2);
        }
        os.indent(indent_num);
        os << cpp_type << " " << extern_attr_name;
        if (type->getValueAsBit("is_array_")) {
          os << ";\n";
          if (extern_attr_name == "shape") {
            os << "CAFFEAttrs attr_shape(attrs." << extern_attr_name
               << "());\n";
            os << extern_attr_name << "= attr_shape.GetShape();\n";
          } else {
            os << " for (int i = 0; i < attrs." << extern_attr_name
               << "().size(); ++i) {\n";
            os.indent(2 + indent_num);
            os << extern_attr_name << ".push_back(attrs." << extern_attr_name
               << "(i))"
               << ";\n";
          }
          os.indent(indent_num);
          os << "}\n";
        } else {
          os << " = "
             << "attrs." << extern_attr_name << "();\n";
        }
        SetAttribute(sn_attr_name, extern_attr_name, spatial_only, index,
                     indent_num, os);
        if (optional_attrs.count(extern_attr_name) != 0) {
          os << "  }";
        }
        os.indent(indent_num);
        os << "\n";
      } else {
        // no need to mapping attribute, such as CMP inst
        os.indent(indent_num);
        os << "inst->Set" << tablegen::Attr::SetAccessName(sn_attr_name) << "("
           << default_value << ");\n";
      }
    }
  }
}

} // namespace tablegen

void EmitRegisterOp(const llvm::RecordKeeper& records, llvm::raw_ostream& os) {
  tablegen::Converter(records).EmitRegisterOp(os);
}
void EmitConverterDecl(const llvm::RecordKeeper& records,
                       llvm::raw_ostream& os) {
  tablegen::Converter(records).EmitConverterDecl(os);
}
void EmitConverterDef(const llvm::RecordKeeper& records,
                      llvm::raw_ostream& os) {
  tablegen::Converter(records).EmitConverterDef(os);
}

} // namespace halo
