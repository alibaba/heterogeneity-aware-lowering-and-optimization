// RUN: %cxx %s -o %t %flags %include %link
// RUN: %t 2>&1| FileCheck %s

#include "halo/lib/ir/ir_builder.h"
#include "halo/lib/ir/values.h"
#include "halo/lib/pass/pass_manager.h"

using namespace halo;

void build() {
  GlobalContext ctx;
  Module m(ctx, "test_module");

  FunctionBuilder func_builder(&m);

  Function* func = func_builder.CreateFunction("func");

  Type ty(DataType::FLOAT32, {3});

  ArgumentBuilder arg_builder(func);
  auto input = arg_builder.CreateArgument("input", ty);

  BasicBlockBuilder bb_builder(func);
  BasicBlock* bb = bb_builder.CreateBasicBlock("bb0");

  IRBuilder ir_builder(bb);

  Instruction* relu0 = ir_builder.CreateRelu("relu0", *input);
  ir_builder.CreateReturn("ret", *relu0);

  // simulate the driver's argv[0] by reading from env var.
  ctx.SetBasePath(getenv("HALO_BASE_PATH"));

  PassManager pm(ctx);
  pm.AddTypeLegalizerPass();
  pm.AddGenericLLVMIRCodeGenPass();
  pm.AddGenericLLVMIRWriterPass(std::cout, false);

  pm.Run(&m);

  // CHECK: %1 = fcmp ogt <3 x float> %0, zeroinitializer
  // clang-format off
  // CHECK-NEXT: %2 = select <3 x i1> %1, <3 x float> %0, <3 x float> zeroinitializer
  // clang-format on
  // CHECK-NEXT: store <3 x float> %2, <3 x float>* %out_relu0
}

int main() { build(); }