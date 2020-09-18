// RUN: %cxx %s -o %t %flags %include %link
// RUN: %t 2>&1| FileCheck %s

#include "halo/lib/ir/ir_builder.h"
#include "halo/lib/ir/values.h"
#include "halo/lib/pass/pass_manager.h"
#include "halo/lib/transforms/type_legalizer.h"

using namespace halo;

void build() {
  GlobalContext ctx;
  Module m(ctx, "test_module");

  FunctionBuilder func_builder(&m);

  Function* func = func_builder.CreateFunction("func");

  BasicBlockBuilder bb_builder(func);
  BasicBlock* bb = bb_builder.CreateBasicBlock("bb0");

  std::vector<float> w0{1.0, 2.0, 3.0};
  std::vector<float> w1{4.0, 5.0, 6.0};

  ConstantBuilder c_builder(func);
  auto c0 =
      c_builder.CreateConstant("w0", Type(DataType::FLOAT32, {3}), w0.data());
  auto c1 = c_builder.CreateConstant("w1", Type(DataType::FLOAT32, {3}), w1);

  IRBuilder ir_builder(bb);

  Instruction* add0 = ir_builder.CreateMul("add0", *c0, *c1);
  Instruction* add1 = ir_builder.CreateAdd("add1", *add0, *add0);

  PassManager pm(ctx);
  pm.AddPass<TypeLegalizer>();

  pm.Dump();
  // CHECK: FunctionPassManager
  // CHECK: BasicBlockPassManager
  // CHECK: Type Legalizer

  pm.Run(&m);

  m.Dump();
  // clang-format off
  // CHECK: Module: test_module
  // CHECK: Function: func()
  // CHECK: Constant w0([FLOAT32: 3]) = [1, 2, 3]
  // CHECK: Constant w1([FLOAT32: 3]) = [4, 5, 6]
  // CHECK: BasicBlock: bb0
  // CHECK-NEXT: Inst: add0([FLOAT32: 3]) = mul(<w0, 0>:[FLOAT32: 3], <w1, 0>:[FLOAT32: 3])
  // CHECK-NEXT: Inst: add1([FLOAT32: 3]) = add(<add0, 0>:[FLOAT32: 3], <add0, 0>:[FLOAT32: 3])
  // clang-format on
}

int main() { build(); }