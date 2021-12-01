// RUN: %cxx %s -o %t %flags %include %link
// RUN: %t 2>&1| FileCheck %s

#include "halo/lib/ir/ir_builder.h"

using namespace halo;

void build() {
  GlobalContext ctx;
  Module m(ctx, "test_module");

  FunctionBuilder func_builder(&m);

  Function* func = func_builder.CreateFunction("func");

  ArgumentBuilder arg_builder(func);
  arg_builder.CreateArgument(
      "pl0", Type(DataType::FLOAT32, std::vector<int64_t>{3, 4, 5}));

  Argument* arg1 = arg_builder.CreateArgument(
      "pl1", Type(DataType::INT32, std::vector<int64_t>{1, 2}));

  BasicBlockBuilder bb_builder(func);
  BasicBlock* bb = bb_builder.CreateBasicBlock("bb0");

  std::vector<float> w0{1.0, 2.0, 3.0};
  std::vector<float> w1{4.0, 5.0, 6.0, 7.0};

  ConstantBuilder c_builder(func);
  c_builder.CreateConstant("w0", Type(DataType::FLOAT32, {3}), w0.data());
  c_builder.CreateConstant("w1", Type(DataType::FLOAT32, {4}), w1);

  IRBuilder ir_builder(bb);

  Instruction* add0 = ir_builder.CreateMul("add0", {});
  Instruction* add2 = ir_builder.CreateAdd("add2", {});

  // Create and insert mul0 before add2
  ir_builder.SetInsertBefore(add0);
  auto mul0 = ir_builder.CreateMul("mul0", *add0, *add0);

  // Create and insert sub1 after mul0
  ir_builder.SetInsertAfter(mul0);
  auto sub1 = ir_builder.CreateSub("sub1", *add2, *add2);

  ir_builder.SetInsertAfter(add2);
  ir_builder.CreateReturn("ret", *sub1);

  m.Dump();

  // CHECK: Module: test_module
  // CHECK: Function: func(pl0[FLOAT32: 3x4x5], pl1[INT32: 1x2])
  // CHECK: Constant w0([FLOAT32: 3]) = [1, 2, 3]
  // CHECK: Constant w1([FLOAT32: 4]) = [4, 5, 6, 7]
  // CHECK: BasicBlock: bb0
  // CHECK-NEXT: Inst: mul0({{.*}}) = mul(<add0, 0>:{{.*}}, <add0, 0>:{{.*}})
  // CHECK-NEXT: Inst: sub1({{.*}}) = sub(<add2, 0>:{{.*}}, <add2, 0>:{{.*}})
  // CHECK-NEXT: Inst: add0({{.*}}) = mul()
  // CHECK-NEXT: Inst: add2({{.*}}) = add()
  // CHECK-NEXT: Inst: ret({{.*}}) = return(<sub1, 0>:{{.*}})
}

int main() { build(); }