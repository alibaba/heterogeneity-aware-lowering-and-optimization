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
  auto jump0 = ir_builder0.CreateJump("jump0", *c_condition);

  BasicBlock* target_block = bb_builder.CreateBasicBlock("target_block");
  IRBuilder ir_builder1(target_block);
  ArgumentBuilder arg_builder(func);
  Argument* arg0 =
      arg_builder.CreateArgument("arg0", Type(DataType::FLOAT32, {6}));
  std::vector<float> input_data{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  auto c_input_data = c_builder.CreateConstant(
      "input_data", Type(DataType::FLOAT32, {6}), input_data.data());
  Instruction* add0 = ir_builder1.CreateAdd("add0", *c_input_data, *arg0);
  jump0->SetTarget(target_block);
  ir_builder1.CreateReturn("ret", std::vector<Def>{*add0});

  ir_builder0.CreateReturn("ret", std::vector<Def>{*jump0});

  PassManager pm(ctx);
  pm.AddPass<TypeLegalizer>();
  pm.Run(&m);
  m.Dump();

  // clang-format off
// CHECK: Module: test_module
// CHECK: Function: func(arg0[FLOAT32: 6])
// CHECK: Constant condition([BOOL: 1]) = [true]
// CHECK: Constant input_data([FLOAT32: 6]) = [1, 2, 3, 4, 5, 6]
// CHECK: BasicBlock: bb0
// CHECK: Inst: jump0([BOOL: 1]) = jump(<condition, 0>:[BOOL: 1]) {Attrs: <target: {{.*}}>}
// CHECK: Inst: ret() = return(<jump0, 0>:[BOOL: 1])
// CHECK: BasicBlock: target_block
// CHECK: Inst: add0([FLOAT32: 6]) = add(<input_data, 0>:[FLOAT32: 6], <arg0, 0>:[FLOAT32: 6])
// CHECK: Inst: ret() = return(<add0, 0>:[FLOAT32: 6])
  // clang-format on
}

int main() {
  build();
  return 0;
}