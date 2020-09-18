// RUN: %cxx %s -o %t %flags %include %link
// RUN: %t 2>&1| FileCheck %s

#include "halo/lib/ir/ir_builder.h"
#include "halo/lib/ir/values.h"
#include "halo/lib/pass/pass_manager.h"
#include "halo/lib/pass/verifier.h"
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

  std::vector<int> w2{7, 8, 9};
  auto c2 =
      c_builder.CreateConstant("w2", Type(DataType::INT32, {3}), w2.data());
  Instruction* add1 = ir_builder.CreateAdd("add1", *add0, *c2);

  PassManager pm(ctx);
  pm.AddPass<VerifierPass>(true);
  pm.AddPass<TypeLegalizer>();
  pm.AddPass<VerifierPass>(false);

  pm.Dump();
  // CHECK: FunctionPassManager
  // CHECK: Verifier
  // CHECK: FunctionPassManager
  // CHECK: BasicBlockPassManager
  // CHECK: Type Legalizer
  // CHECK: FunctionPassManager
  // CHECK: Verifier

  pm.Run(&m);
  // CHECK: type match is expected at operand 1 and 0.
  // CHECK: Inst: add1 = add(<add0, 0>, <w2, 0>)
  // CHECK: IR is broken.
  // XFAIL: *
}

int main() { build(); }