// RUN: %cxx %s -o %t %flags %include %link -DBUILD_IR
// RUN: %t > %t.obj
// RUN: %cxx %s %t.obj -o %t2
// RUN: %t2  2>&1| FileCheck %s

#ifdef BUILD_IR
#include "halo/lib/ir/ir_builder.h"
#include "halo/lib/ir/values.h"
#include "halo/lib/pass/pass_manager.h"

using namespace halo;

void Build() {
  GlobalContext ctx;
  Module m(ctx, "test_module");

  FunctionBuilder func_builder(&m);

  Function* func = func_builder.CreateFunction("func");

  ArgumentBuilder arg_builder(func);
  auto input =
      arg_builder.CreateArgument("input", Type{DataType::FLOAT32, {2, 1}});

  BasicBlockBuilder bb_builder(func);
  BasicBlock* bb = bb_builder.CreateBasicBlock("bb0");

  std::vector<float> w0{1.0, 2.0};

  ConstantBuilder c_builder(func);
  auto w = c_builder.CreateConstant("w0", Type{DataType::FLOAT32, {1, 2}},
                                    w0.data());

  IRBuilder ir_builder(bb);

  Instruction* add = ir_builder.CreateAdd("add", *input, *w);
  ir_builder.CreateReturn("ret", *add);

  // simulate the driver's argv[0] by reading from env var.
  ctx.SetBasePath(getenv("HALO_BASE_PATH"));

  PassManager pm(ctx);
  pm.AddTypeLegalizerPass();
  pm.AddX86LLVMIRCodeGenPass();
  pm.AddX86BinaryWriterPass(std::cout);

  pm.Run(&m);
}

int main() { Build(); }

#else

#include <stdio.h>

extern "C" {
extern void func(float* input, float* output);
}

int main() {
  float input[] = {5.0f, 4.0f};
  float output[4];
  func(input, output);
  // CHECK: 6.000000 7.000000 5.000000 6.000000
  for (int i = 0; i < 4; ++i) {
    printf("%f ", output[i]);
  }
}
#endif