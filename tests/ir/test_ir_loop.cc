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
  std::vector<int64_t> loop_cnt{2};
  auto c_loop_cnt = c_builder.CreateConstant(
      "loop_cnt", Type(DataType::INT64, {1}), loop_cnt.data());
  std::vector<int8_t> loop_range{1};
  auto c_loop_range = c_builder.CreateConstant(
      "loop_range", Type(DataType::BOOL, {1}), loop_range.data());
  std::vector<float> input_data{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  auto c_input_data = c_builder.CreateConstant(
      "input_data", Type(DataType::FLOAT32, {2, 3}), input_data.data());

  BasicBlockBuilder bb_builder(func);
  BasicBlock* bb0 = bb_builder.CreateBasicBlock("bb0");
  IRBuilder ir_builder0(bb0);
  auto loop0 = ir_builder0.CreateLoop("loop0", *c_loop_cnt, *c_loop_range,
                                      *c_input_data);

  BasicBlock* loop_body = bb_builder.CreateBasicBlock("loop_body");
  IRBuilder ir_builder1(loop_body);
  ArgumentBuilder arg_builder(func);
  Argument* arg_loop_cnt =
      arg_builder.CreateArgument("arg_loop_cnt", Type(DataType::INT64, {1}));
  Argument* arg_loop_range =
      arg_builder.CreateArgument("arg_loop_range", Type(DataType::BOOL, {1}));
  Argument* arg_input_data = arg_builder.CreateArgument(
      "arg_input_data", Type(DataType::FLOAT32, {2, 3}));
  Instruction* add =
      ir_builder1.CreateAdd("add", *arg_loop_cnt, *arg_input_data);
  loop0->SetBody(loop_body);

  ir_builder0.CreateReturn("ret", std::vector<Def>{*loop0});

  PassManager pm(ctx);
  pm.AddPass<TypeLegalizer>();
  pm.Run(&m);
  m.Dump();

  // clang-format off
// CHECK: Module: test_module
// CHECK: Function: func(arg_loop_cnt[INT64: 1], arg_loop_range[BOOL: 1], arg_input_data[FLOAT32: 2x3])
// CHECK: Constant loop_cnt([INT64: 1]) = [2]
// CHECK: Constant loop_range([BOOL: 1]) = [true]
// CHECK: Constant input_data([FLOAT32: 2x3]) = [1, 2, 3, 4, 5, 6]
// CHECK: BasicBlock: bb0
// CHECK: Inst: loop0([INT64: 1]) = loop(<loop_cnt, 0>:[INT64: 1], <loop_range, 0>:[BOOL: 1], <input_data, 0>:[FLOAT32: 2x3]) {Attrs: <body: {{.*}}>}
// CHECK: Inst: ret() = return(<loop0, 0>:[INT64: 1])
// CHECK: BasicBlock: loop_body
// CHECK: Inst: add([INT64: 2x3]) = add(<arg_loop_cnt, 0>:[INT64: 1], <arg_input_data, 0>:[FLOAT32: 2x3])
  // clang-format on
}

int main() {
  build();
  return 0;
}