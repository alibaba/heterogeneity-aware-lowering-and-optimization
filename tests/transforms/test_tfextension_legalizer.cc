// RUN: %cxx %s -o %t %flags %include %link
// RUN: %t 2>&1| FileCheck %s

#include "halo/lib/ir/ir_builder.h"
#include "halo/lib/ir/values.h"
#include "halo/lib/pass/pass_manager.h"
#include "halo/lib/transforms/dce.h"
#include "halo/lib/transforms/tfextension_legalizer.h"

using namespace halo;

void build() {
  GlobalContext ctx;
  Module m(ctx, "test_module");

  FunctionBuilder func_builder(&m);

  Function* func = func_builder.CreateFunction("func");

  BasicBlockBuilder bb_builder(func);
  BasicBlock* bb = bb_builder.CreateBasicBlock("bb0");

  std::vector<float> w0{1.0, 2.0, 3.0, 4.0};
  std::vector<int> padding_amt{1, 1, 1, 1};

  ConstantBuilder c_builder(func);
  auto c0 = c_builder.CreateConstant("w0", Type(DataType::FLOAT32, {2, 2}),
                                     w0.data());
  auto c1 = c_builder.CreateConstant("amt", Type(DataType::INT32, {2, 2}),
                                     padding_amt.data());

  IRBuilder ir_builder(bb);

  Instruction* tf =
      ir_builder.CreateTFExtension("tf", std::vector<Def>{*c0}, 1, "Identity");
  Instruction* pad = ir_builder.CreatePad("pad", std::vector<Def>{*tf, *c1});
  ir_builder.CreateReturn("ret", *pad);

  PassManager pm(ctx);
  pm.AddPass<halo::TFExtensionLegalizer>(true);
  pm.AddPass<halo::DCE>();
  pm.Run(&m);

  m.Dump();

  // clang-format off
  // CHECK: Module: test_module
  // CHECK: Function: func()

  // CHECK: BasicBlock: bb0
  // CHECK-NOT: tf
  // CHECK: pad({{.*}}) = pad(<w0, 0>:[FLOAT32: 2x2], <amt, 0>:[INT32: 2x2])
  // clang-format on
}

int main() { build(); }