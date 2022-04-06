//===- extension_instructions.h ---------------------------------*- C++ -*-===//
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

#ifndef HALO_LIB_IR_CUSTOM_INST_H_
#define HALO_LIB_IR_CUSTOM_INST_H_

#include <set>
#include <unordered_map>

#include "halo/lib/ir/instruction.h"

namespace halo {

const int kEightBits = 8;
const int kSixteenBits = 16;

/// This is a base Extension instruction class
/// It defines either a user op or an op defined in other frameworks yet
/// not in Halo IR.
class ExtensionInst : public Instruction {
 public:
  enum class ExtensionKind {
    kExtension_CAFFE,
    kExtension_TENSORFLOW,
    kExtension_ONNX,
    kExtension_TFLITE,
    kExtension_CUSTOM,
  };
  /// Define a custom instruction, its semantics is defined by
  /// a user opcode or opcode in other frameworks.
  explicit ExtensionInst(GlobalContext& context, const std::string& name,
                         const std::vector<Def>& operands, int num_outs,
                         const std::string& opname, OpCode opcode);
  /// Return the extension opcode name
  const std::string& GetOpName() const noexcept { return opname_; }
  /// Set the name.
  void SetOpName(const std::string& name) noexcept { opname_ = name; }
  void PrintOpcode(std::ostream& os) const final { os << opname_; }
  bool IsOperandOptional(size_t idx) const noexcept override;
  void MarkOperandOptional(size_t idx) noexcept;
  virtual ExtensionKind GetExtensionKind() const noexcept = 0;
  static inline bool Classof(const IRObject* obj) {
    if (!Instruction::Classof(obj)) {
      return false;
    }
    const Instruction* inst = Downcast<const Instruction>(obj);
    return inst->GetOpCode() == OpCode::EXTENSION;
  }

 private:
  std::string opname_;
  std::set<int> optional_args_;
};

/// This defines a tensorflow op which cannot be one-on-one mapped to
/// Halo IR.
class TFExtensionInst final : public ExtensionInst {
 public:
  using NameToTFOpMap = std::unordered_map<std::string, TFExtOpCode>;

  explicit TFExtensionInst(GlobalContext& context, const std::string& name,
                           const std::vector<Def>& operands, int num_outs,
                           const std::string& opname);
  /// Return the TF extension opcode.
  TFExtOpCode GetExtOpCode() const noexcept { return ext_op_code_; }
  ExtensionKind GetExtensionKind() const noexcept override {
    return ExtensionKind::kExtension_TENSORFLOW;
  }
  std::unique_ptr<Instruction> Clone() const override;

  static inline bool Classof(const IRObject* obj) {
    if (!ExtensionInst::Classof(obj)) {
      return false;
    }
    const ExtensionInst* inst = Downcast<const ExtensionInst>(obj);
    return inst->GetExtensionKind() ==
           ExtensionInst::ExtensionKind::kExtension_TENSORFLOW;
  }

 private:
  // A string name to TF extension opcode map.
  static const NameToTFOpMap TFMap;
  // TF extension opcode.
  TFExtOpCode ext_op_code_ = TFExtOpCode::UNSUPPORTED;
};

/// This defines a user op.
class CustomInst final : public ExtensionInst {
 public:
  explicit CustomInst(GlobalContext& context, const std::string& name,
                      const std::vector<Def>& operands, int num_outs,
                      const std::string& opname)
      : ExtensionInst(context, name, operands, num_outs, "custom_" + opname,
                      OpCode::CUSTOM) {}
  CustomExtOpCode GetExtOpCode() const noexcept { return ext_op_code_; }
  ExtensionKind GetExtensionKind() const noexcept override {
    return ExtensionKind::kExtension_CUSTOM;
  }
  std::unique_ptr<Instruction> Clone() const override;

 private:
  CustomExtOpCode ext_op_code_ = CustomExtOpCode::UNSUPPORTED;
};

/// This class represents a dummy instruction.
class DummyInst final : public ExtensionInst {
 public:
  explicit DummyInst(GlobalContext& context, const std::string& name,
                     const std::vector<Def>& operands, int num_outs,
                     const std::string& opname)
      : ExtensionInst(context, name, operands, num_outs, "dummy_" + opname,
                      OpCode::INVALID) {}
  CustomExtOpCode GetExtOpCode() const noexcept { return ext_op_code_; }
  ExtensionKind GetExtensionKind() const noexcept override {
    return ExtensionKind::kExtension_CUSTOM;
  }
  std::unique_ptr<Instruction> Clone() const override;

 private:
  CustomExtOpCode ext_op_code_ = CustomExtOpCode::UNSUPPORTED;
};

/// This defines a onnx op which cannot be one-on-one mapped to
/// Halo IR.
class ONNXExtensionInst final : public ExtensionInst {
 public:
  using NameToOpMap = std::unordered_map<std::string, ONNXExtOpCode>;

  explicit ONNXExtensionInst(GlobalContext& context, const std::string& name,
                             const std::vector<Def>& operands, int num_outs,
                             const std::string& opname);
  /// Return the extension opcode.
  ONNXExtOpCode GetExtOpCode() const noexcept { return ext_op_code_; }
  ExtensionKind GetExtensionKind() const noexcept override {
    return ExtensionKind::kExtension_ONNX;
  }
  std::unique_ptr<Instruction> Clone() const override;

  static inline bool Classof(const IRObject* obj) {
    if (!ExtensionInst::Classof(obj)) {
      return false;
    }
    const ExtensionInst* inst = Downcast<const ExtensionInst>(obj);
    return inst->GetExtensionKind() ==
           ExtensionInst::ExtensionKind::kExtension_ONNX;
  }

 private:
  // A string name to extension opcode map.
  static const NameToOpMap ONNXMap;
  // ONNX extension opcode.
  ONNXExtOpCode ext_op_code_ = ONNXExtOpCode::UNSUPPORTED;
};

/// This defines a tflite op which cannot be one-on-one mapped to
/// Halo IR.
class TFLITEExtensionInst final : public ExtensionInst {
 public:
  using NameToOpMap = std::unordered_map<std::string, TFLITEExtOpCode>;

  explicit TFLITEExtensionInst(GlobalContext& context, const std::string& name,
                               const std::vector<Def>& operands, int num_outs,
                               const std::string& opname);
  /// Return the extension opcode.
  TFLITEExtOpCode GetExtOpCode() const noexcept { return ext_op_code_; }
  ExtensionKind GetExtensionKind() const noexcept override {
    return ExtensionKind::kExtension_TFLITE;
  }
  std::unique_ptr<Instruction> Clone() const override;

  static inline bool Classof(const IRObject* obj) {
    if (!ExtensionInst::Classof(obj)) {
      return false;
    }
    const ExtensionInst* inst = Downcast<const ExtensionInst>(obj);
    return inst->GetExtensionKind() ==
           ExtensionInst::ExtensionKind::kExtension_TFLITE;
  }

 private:
  // A string name to extension opcode map.
  static const NameToOpMap TFLITEMap;
  // TFLITE extension opcode.
  TFLITEExtOpCode ext_op_code_ = TFLITEExtOpCode::UNSUPPORTED;
};

/// This defines a onnx op which cannot be one-on-one mapped to
/// Halo IR.
class CAFFEExtensionInst final : public ExtensionInst {
 public:
  using NameToOpMap = std::unordered_map<std::string, CAFFEExtOpCode>;

  explicit CAFFEExtensionInst(GlobalContext& context, const std::string& name,
                              const std::vector<Def>& operands, int num_outs,
                              const std::string& opname);
  /// Return the extension opcode.
  CAFFEExtOpCode GetExtOpCode() const noexcept { return ext_op_code_; }
  ExtensionKind GetExtensionKind() const noexcept override {
    return ExtensionKind::kExtension_CAFFE;
  }
  std::unique_ptr<Instruction> Clone() const override;

  static inline bool Classof(const IRObject* obj) {
    if (!ExtensionInst::Classof(obj)) {
      return false;
    }
    const ExtensionInst* inst = Downcast<const ExtensionInst>(obj);
    return inst->GetExtensionKind() ==
           ExtensionInst::ExtensionKind::kExtension_CAFFE;
  }

 private:
  // A string name to extension opcode map.
  static const NameToOpMap CAFFEMap;
  // CAFFE extension opcode.
  CAFFEExtOpCode ext_op_code_ = CAFFEExtOpCode::UNSUPPORTED;
};

} // end namespace halo

#endif // HALO_LIB_IR_CUSTOM_INST_H_