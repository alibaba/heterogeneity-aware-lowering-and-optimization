//===- doc_emitter.cc -------------------------------------------*- C++ -*-===//
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

#include "inst.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

namespace halo {

namespace tablegen {

/// A set of instructions in the category
class Category {
 public:
  Category() = default;
  explicit Category(const std::string& name) : name_(name) {}
  /// Append Inst to the list
  void AddInst(Inst one) { insts_.push_back(std::move(one)); }
  /// Return Inst list
  std::vector<Inst>& GetInsts() { return insts_; }
  /// Return category name
  const std::string& GetName() const { return name_; }

 private:
  // Inst list belong to the category
  std::vector<Inst> insts_;
  // Parent catgory
  Category* parent_ = nullptr;
  // Category name
  std::string name_;
};

/// Document class
class Doc {
 public:
  Doc(const llvm::RecordKeeper& records, llvm::raw_ostream& os);
  /// Emit table of contents
  void EmitTableOfContents();
  /// Emit contents
  void EmitBody();
  /// Main entry to emic doc
  void Run();
  /// Generate a normalized category name
  /// e.g., cat_abc_def in td becomes abc.def
  static std::string NormalizeCatName(llvm::StringRef key);
  /// Generate string to create toc-like link in .md
  static std::string CreateMDLink(const std::string& name, bool is_toc);

 private:
  llvm::raw_ostream& os_;
  // Category list in the doc
  std::vector<Category> all_cats_;
};

Doc::Doc(const llvm::RecordKeeper& records, llvm::raw_ostream& os) : os_(os) {
  auto inst_records = records.getAllDerivedDefinitions("Inst");
  auto cat_records = records.getAllDerivedDefinitions("Category");
  std::unordered_map<llvm::Record*, size_t> record_to_cat_map;
  for (auto r : cat_records) {
    std::string cat_name = Doc::NormalizeCatName(r->getName());
    all_cats_.emplace_back(cat_name);
    record_to_cat_map.emplace(r, all_cats_.size() - 1);
  }
  for (auto r : inst_records) {
    llvm::Record* cat_key = r->getValueAsDef("cat_");
    Category& cat = all_cats_[record_to_cat_map[cat_key]];
    cat.AddInst(Inst(r, os));
  }
}

void Doc::Run() {
  os_ << "# Halo IR  \n";
  os_ << "---\n";
  EmitTableOfContents();
  os_ << "---\n";
  EmitBody();
}

void Doc::EmitTableOfContents() {
  int i = 0;
  for (auto& cat : all_cats_) {
    os_ << "## " << ++i << ". ";
    os_ << Doc::CreateMDLink(cat.GetName(), true) << "  \n";
    std::vector<Inst>& insts = cat.GetInsts();
    for (auto& inst : insts) {
      os_ << "* ";
      os_ << Doc::CreateMDLink(inst.GetOpName(), true) << "  \n";
    }
    os_ << "\n";
  }
}

void Doc::EmitBody() {
  for (auto& cat : all_cats_) {
    os_ << Doc::CreateMDLink(cat.GetName(), false) << "\n";
    os_ << "# " << cat.GetName() << "  \n";
    os_ << "---\n";
    std::vector<Inst>& insts = cat.GetInsts();
    for (auto& inst : insts) {
      os_ << Doc::CreateMDLink(inst.GetOpName(), false) << "\n";
      inst.EmitDoc();
    }
  }
}

std::string Doc::CreateMDLink(const std::string& name, bool is_toc) {
  std::string result;
  if (is_toc) {
    result = "[" + name + "](#" + name + ")";
  } else {
    result = "<a id=\"" + name + "\"></a>";
  }
  return result;
}

std::string Doc::NormalizeCatName(const llvm::StringRef key) {
  // cat def name starts with 'cat_'
  auto name = key.drop_front(4);
  llvm::SmallVector<llvm::StringRef, 2> splits;
  name.split(splits, '_', -1, false);
  std::string result;
  size_t i = 0;
  for (auto one : splits) {
    result += one.str();
    if (++i != splits.size()) {
      result += ".";
    }
  }
  return result;
}

} // end namespace tablegen

void EmitDoc(const llvm::RecordKeeper& records, llvm::raw_ostream& os) {
  tablegen::Doc(records, os).Run();
}

} // end namespace halo