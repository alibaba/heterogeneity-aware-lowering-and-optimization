// RUN: %cxx %s -o %t %flags %include %link
// RUN: %t 2>&1| FileCheck %s

#include "halo/lib/ir/ir_builder.h"
#include "halo/lib/pass/pass_manager.h"
#include "halo/lib/target/generic_cxx/generic_cxx_codegen.h"
#include "halo/lib/transforms/type_legalizer.h"

using namespace halo;

void build() {
  GlobalContext ctx;
  Module m(ctx, "test_module");

  FunctionBuilder func_builder(&m);
  Function* func = func_builder.CreateFunction("func");

  ConstantBuilder c_builder(func);
  std::vector<int8_t> condition{1};
  auto c_condition = c_builder.CreateConstant(
      "condition", Type(DataType::BOOL, {1}), condition.data());

  BasicBlockBuilder bb_builder(func);
  BasicBlock* bb0 = bb_builder.CreateBasicBlock("bb0");
  IRBuilder ir_builder0(bb0);
  auto if0 = ir_builder0.CreateIf("if0", *c_condition);

  BasicBlock* else_branch = bb_builder.CreateBasicBlock("else_branch");
  IRBuilder ir_builder1(else_branch);
  ArgumentBuilder arg_builder(func);
  Argument* arg0 =
      arg_builder.CreateArgument("arg0", Type(DataType::FLOAT32, {6}));
  std::vector<float> input_data{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  auto c_input_data = c_builder.CreateConstant(
      "input_data", Type(DataType::FLOAT32, {6}), input_data.data());
  Instruction* add0 = ir_builder1.CreateAdd("add0", *c_input_data, *arg0);
  if0->SetElseBranch(else_branch);
  ir_builder1.CreateReturn("ret", std::vector<Def>{*add0});

  BasicBlock* then_branch = bb_builder.CreateBasicBlock("then_branch");
  IRBuilder ir_builder2(then_branch);
  ArgumentBuilder arg_builder1(func);
  Argument* arg1 =
      arg_builder1.CreateArgument("arg1", Type(DataType::FLOAT32, {6}));
  std::vector<float> input_data1{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  auto c_input_data1 = c_builder.CreateConstant(
      "input_data1", Type(DataType::FLOAT32, {6}), input_data1.data());
  Instruction* add1 = ir_builder2.CreateAdd("add1", *c_input_data1, *arg1);
  if0->SetThenBranch(then_branch);
  ir_builder2.CreateReturn("ret", std::vector<Def>{*add1});

  ir_builder0.CreateReturn("ret", std::vector<Def>{*if0});

  PassManager pm(ctx);
  pm.AddPass<TypeLegalizer>();
  pm.Run(&m);
  m.Dump();

  // clang-format off
// CHECK: Module: test_module
// CHECK: Function: func(arg0[FLOAT32: 6], arg1[FLOAT32: 6])
// CHECK: Constant condition([BOOL: 1]) = [true]
// CHECK: Constant input_data([FLOAT32: 6]) = [1, 2, 3, 4, 5, 6]
// CHECK: Constant input_data1([FLOAT32: 6]) = [1, 2, 3, 4, 5, 6]
// CHECK: BasicBlock: bb0
// CHECK: Inst: if0([FLOAT32: 6]) = if(<condition, 0>:[BOOL: 1]) {Attrs: <else_branch: {{.*}}>, <then_branch: {{.*}}>}
// CHECK: Inst: ret([FLOAT32: 6]) = return(<if0, 0>:[FLOAT32: 6])
// CHECK: BasicBlock: else_branch
// CHECK: Inst: add0([FLOAT32: 6]) = add(<input_data, 0>:[FLOAT32: 6], <arg0, 0>:[FLOAT32: 6])
// CHECK: Inst: ret([FLOAT32: 6]) = return(<add0, 0>:[FLOAT32: 6])
// CHECK: BasicBlock: then_branch
// CHECK: Inst: add1([FLOAT32: 6]) = add(<input_data1, 0>:[FLOAT32: 6], <arg1, 0>:[FLOAT32: 6])
// CHECK: Inst: ret([FLOAT32: 6]) = return(<add1, 0>:[FLOAT32: 6])
  // clang-format on
}

int main() {
  build();
  return 0;
}