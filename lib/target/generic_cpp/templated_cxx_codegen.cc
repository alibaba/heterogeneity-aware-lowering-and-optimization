//===- templated_cxx_codegen.cc -------------------------------------------===//
//
// Copyright (C) 2019-2021 Alibaba Group Holding Limited.
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

#include "halo/lib/target/generic_cxx/templated_cxx_codegen.h"

#include <clang/AST/ASTConsumer.h>
#include <clang/AST/RecursiveASTVisitor.h>
#include <clang/CodeGen/CodeGenAction.h>
#include <clang/Frontend/ASTConsumers.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/FrontendActions.h>
#include <clang/Rewrite/Core/Rewriter.h>
#include <clang/Rewrite/Frontend/FrontendActions.h>
#include <clang/Rewrite/Frontend/Rewriters.h>
#include <clang/StaticAnalyzer/Frontend/FrontendActions.h>
#include <clang/Tooling/CompilationDatabase.h>
#include <clang/Tooling/Core/Replacement.h>
#include <clang/Tooling/Tooling.h>
#include <llvm/ADT/APInt.h>
#include <llvm/Support/VirtualFileSystem.h>
#include <llvm/Support/raw_os_ostream.h>

#include <cstddef>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>

#include "halo/lib/target/codegen.h"
#include "halo/lib/target/codegen_object.h"
#include "halo/lib/target/generic_cxx/generic_cxx_codegen.h"

struct CXXModelInfo {
  explicit CXXModelInfo(const std::ostringstream& model_code)
      : ModelCode(model_code) {}
  const std::ostringstream& ModelCode;
  std::vector<std::string> InputIds;
  std::vector<halo::Type> InputTypes;
  std::vector<std::string> OutputIds;
  std::vector<halo::Type> OutputTypes;
};

class Visitor : public clang::RecursiveASTVisitor<Visitor> {
  clang::ASTContext& ast_ctx_;
  clang::Rewriter rewriter_;
  std::ostream& output_;
  const CXXModelInfo& info_;

 public:
  explicit Visitor(clang::CompilerInstance* ci, std::ostream& output,
                   const CXXModelInfo& info)
      : ast_ctx_(ci->getASTContext()), output_(output), info_(info) {
    rewriter_.setSourceMgr(ast_ctx_.getSourceManager(), ast_ctx_.getLangOpts());
  }

  bool HandleValueDecl(clang::VarDecl* vd) {
    clang::AnnotateAttr* attr = vd->getAttr<clang::AnnotateAttr>();
    if (attr == nullptr) {
      return true;
    }
    const auto& annotate = attr->getAnnotation();
    if (!annotate.startswith("halo_")) {
      return true;
    }
    vd->dropAttrs();
    clang::SourceRange range{vd->getBeginLoc(), vd->getEndLoc()};
    auto no_loc = clang::SourceLocation();
    constexpr int width = 32;
    auto get_int_literal = [this, &no_loc](uint64_t x) {
      return clang::IntegerLiteral::Create(ast_ctx_, llvm::APInt(width, x),
                                           ast_ctx_.IntTy, no_loc);
    };

    if (annotate == "halo_nr_inputs" || annotate == "halo_nr_outputs") {
      auto val = annotate == "halo_nr_inputs" ? info_.InputIds.size()
                                              : info_.OutputIds.size();
      llvm::APInt x(width, val, false);
      auto n = clang::IntegerLiteral::Create(ast_ctx_, x, vd->getType(),
                                             vd->getBeginLoc());
      vd->setInit(n);
    } else if (annotate == "halo_input_ids" || annotate == "halo_output_ids") {
      const auto& strings =
          annotate == "halo_input_ids" ? info_.InputIds : info_.OutputIds;
      auto ty = ast_ctx_.getConstantArrayType(
          ast_ctx_.CharTy, llvm::APInt(width, strings.size()), nullptr,
          clang::ArrayType::ArraySizeModifier::Normal, 0);
      std::vector<clang::Expr*> id_strs;
      id_strs.reserve(strings.size());
      for (auto& name : strings) {
        id_strs.push_back(clang::StringLiteral::Create(
            ast_ctx_, name, clang::StringLiteral::StringKind::Ascii, false, ty,
            clang::SourceLocation()));
      }
      clang::Expr* ids_expr =
          new (ast_ctx_) clang::InitListExpr(ast_ctx_, no_loc, id_strs, no_loc);
      vd->setInit(ids_expr);
    } else if (annotate == "halo_output_sizes") {
      std::vector<clang::Expr*> sizes;
      sizes.reserve(info_.OutputIds.size());
      for (const auto& ty : info_.OutputTypes) {
        sizes.push_back(get_int_literal(ty.GetTotalNumOfElements()));
      }
      clang::Expr* ids_expr =
          new (ast_ctx_) clang::InitListExpr(ast_ctx_, no_loc, sizes, no_loc);

      vd->setInit(ids_expr);
    } else if (annotate == "halo_output_types") {
      std::vector<clang::Expr*> types;
      types.reserve(info_.OutputTypes.size());
      for (const auto& ty : info_.OutputTypes) {
        std::vector<clang::Expr*> odla_ty;
        auto ty_expr = get_int_literal(static_cast<uint64_t>(ty.GetDataType()));
        auto rank = get_int_literal(ty.GetNumOfDims());
        std::vector<clang::Expr*> dims;
        dims.reserve(ty.GetNumOfDims());
        for (uint64_t dim : ty.GetDimSizes()) {
          dims.push_back(get_int_literal(dim));
        }
        clang::Expr* dims_list =
            new (ast_ctx_) clang::InitListExpr(ast_ctx_, no_loc, dims, no_loc);
        clang::Expr* shape_expr = new (ast_ctx_)
            clang::InitListExpr(ast_ctx_, no_loc, {rank, dims_list}, no_loc);
        clang::Expr* type_expr = new (ast_ctx_) clang::InitListExpr(
            ast_ctx_, no_loc, {ty_expr, shape_expr}, no_loc);
        types.push_back(type_expr);
      }
      clang::Expr* types_expr =
          new (ast_ctx_) clang::InitListExpr(ast_ctx_, no_loc, types, no_loc);
      vd->setInit(types_expr);
    }

    std::string txt;
    llvm::raw_string_ostream oss(txt);
    vd->print(oss);
    rewriter_.ReplaceText(range, oss.str());
    return true;
  }

  bool VisitFunctionDecl(clang::FunctionDecl* fd) {
    clang::AnnotateAttr* attr = fd->getAttr<clang::AnnotateAttr>();
    if (attr != nullptr) {
      const auto& annotate = attr->getAnnotation();
      if (annotate == "halo_build_computation") {
        auto loc = ast_ctx_.getFullLoc(fd->getBeginLoc()).getSpellingLoc();
        auto loc_e = ast_ctx_.getFullLoc(fd->getEndLoc()).getSpellingLoc();
        fd->dropAttrs();
        std::string txt;

        txt += "static odla_computation " + fd->getName().str() + "() {\n";
        txt += info_.ModelCode.str();
        txt += "\n};";
        rewriter_.ReplaceText(clang::SourceRange{loc.getLocWithOffset(0),
                                                 loc_e.getLocWithOffset(0)},
                              txt);
      }
    }
    return true;
  }

  bool VisitVarDecl(clang::VarDecl* vd) { return HandleValueDecl(vd); }

  void Emit() {
    llvm::raw_os_ostream ros(output_);
    auto& buf =
        rewriter_.getEditBuffer(rewriter_.getSourceMgr().getMainFileID());
    buf.write(ros);
    ros.flush();
  }
};

class HALOAction : public clang::ASTFrontendAction {
  class HALOConsumer : public clang::ASTConsumer {
   public:
    explicit HALOConsumer(clang::CompilerInstance* ci, std::ostream& output,
                          const CXXModelInfo& info)
        : visitor_(ci, output, info) {}

    void HandleTranslationUnit(clang::ASTContext& ctx) final {
      visitor_.TraverseDecl(ctx.getTranslationUnitDecl());
      visitor_.Emit();
    }

   private:
    Visitor visitor_;
  };

 public:
  explicit HALOAction(std::ostream& output, const CXXModelInfo& info,
                      bool verbose)
      : output_(output), info_(info), verbose_(verbose) {
    dc_ = std::make_unique<clang::IgnoringDiagConsumer>();
  }
  std::unique_ptr<clang::ASTConsumer> CreateASTConsumer(
      clang::CompilerInstance& ci, llvm::StringRef file) override {
    if (!verbose_) {
      ci.getDiagnostics().setClient(dc_.get(), false);
    }
    return std::make_unique<HALOConsumer>(&ci, output_, info_);
  }

 private:
  std::ostream& output_;
  const CXXModelInfo& info_;
  bool verbose_;
  std::unique_ptr<clang::IgnoringDiagConsumer> dc_;
};

class HaloFrontendActionFactory : public clang::tooling::FrontendActionFactory {
 public:
  explicit HaloFrontendActionFactory(std::ostream& output,
                                     const CXXModelInfo& info, bool verbose)
      : output_(output), info_(info), verbose_(verbose) {}
  std::unique_ptr<clang::FrontendAction> create() override {
    auto halo_act = std::make_unique<HALOAction>(output_, info_, verbose_);
    return halo_act;
  }

 private:
  std::ostream& output_;
  const CXXModelInfo& info_;
  bool verbose_;
};

static void EmitWithTemplate(std::ostream& output, const CXXModelInfo& info,
                             const std::string& template_file,
                             const std::string& odla_inc_dir, bool verbose) {
  clang::tooling::FixedCompilationDatabase comp_db("/", {});

  clang::tooling::ClangTool tool(comp_db, {template_file});
  // TODO(unknown): Verify template file
  // tool.mapVirtualFile("/input.cc", halo_template);
  HaloFrontendActionFactory factory(output, info, verbose);
  std::string odla_inc = "-I" + odla_inc_dir;
  if (!odla_inc_dir.empty()) {
    std::cout << odla_inc_dir << std::endl;
    tool.appendArgumentsAdjuster(
        clang::tooling::getInsertArgumentAdjuster(odla_inc.c_str()));
  }
  if (verbose) {
    tool.appendArgumentsAdjuster(
        clang::tooling::getInsertArgumentAdjuster("-v"));
  }

  tool.run(&factory);
}

namespace halo {

TemplatedCXXCodeGen::TemplatedCXXCodeGen(std::ostringstream& os,
                                         std::ostringstream& header_os,
                                         const CXXCodeGenOpts& opts)
    : GenericCXXCodeGen(generic_os_, header_os, std::cout, opts), code_os_(os) {
  emit_banner = false;
}

void TemplatedCXXCodeGen::RunOnFunction(Function& function) {
  generic_os_ << "  odla_computation _comp;\n";
  generic_os_ << "  odla_CreateComputation(&_comp);\n";
  const auto& ctx = function.GetGlobalContext();

  // Declare external data.
  for (auto& constant : function.Constants()) {
    RunOnConstant(*constant, true);
  }

  if (function.empty() || (function.BasicBlocks().size() == 1 &&
                           function.BasicBlocks().front()->empty())) {
    return;
  }

  Instruction* return_inst = function.GetReturnInst();
  HLCHECK(return_inst && "No Return Instruction found");

  // Emit wrappers for arguments.
  for (auto& arg : function.Args()) {
    RunOnArgument(*arg);
  }
  // Emit wrappers for constants.
  for (auto& constant : function.Constants()) {
    RunOnConstant(*constant, false);
  }

  for (auto& bb : function) {
    RunOnBasicBlock(*bb);
  }
  generic_os_ << " return _comp;\n";
  CXXModelInfo info(generic_os_);
  for (auto& arg : function.Args()) {
    info.InputIds.push_back(arg->GetName());
    info.InputTypes.push_back(arg->GetResultType());
  }
  for (auto& op : return_inst->GetOperands()) {
    auto& cv = ir_mapping_[op];
    info.OutputIds.push_back(cv.GetName());
    info.OutputTypes.push_back(op.GetType());
  }
  EmitWithTemplate(code_os_, info, opts_.template_file,
                   ctx.GetODLAIncludePath(), ctx.GetVerbosity() > 0);
}

TemplatedCXXCodeGen::~TemplatedCXXCodeGen() = default;

} // namespace halo
