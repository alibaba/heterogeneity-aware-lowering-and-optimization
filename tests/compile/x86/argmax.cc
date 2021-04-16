// RUN: %cxx %s -o %t %flags %include %link -DBUILD_IR
// RUN: %t > %t.obj
// RUN: %cxx %s %t.obj %flags -static -o %t2
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
      arg_builder.CreateArgument("input", Type{DataType::FLOAT32, {1, 3, 4}});

  BasicBlockBuilder bb_builder(func);
  BasicBlock* bb = bb_builder.CreateBasicBlock("bb0");

  std::vector<int> w0{-1};

  ConstantBuilder c_builder(func);
  auto axis = c_builder.CreateConstant("w0", Type{DataType::INT32}, w0.data());

  IRBuilder ir_builder(bb);

  ArgmaxInst* reduce = ir_builder.CreateArgmax("argmax", *input, *axis);
  reduce->SetKeepDims(false);
  ir_builder.CreateReturn("ret", *reduce);

  // simulate the driver's argv[0] by reading from env var.
  ctx.SetBasePath(getenv("HALO_BASE_PATH"));

  PassManager pm(ctx);
  pm.AddTypeLegalizerPass();
  pm.AddDCEPass();
  pm.AddInstSimplifyPass();
  pm.Run(&m);
  // pm.AddPass<GenericLLVMIRCodeGen>();
  // pm.AddPass<GenericLLVMIRWriter>(std::ref(std::cout), false);
  pm.AddX86LLVMIRCodeGenPass();
  pm.AddX86BinaryWriterPass(std::cout);

  pm.Run(&m);
}

int main() { Build(); }

#else

#include <stdio.h>

extern "C" {
extern void func(float* input, int* output);
}

int main() {
  float input[] = {5.0,  1.0, 20.0, 2.0, 30.0, 1.0,
                   40.0, 2.0, 55.0, 1.0, 30.0, 2.0};
  int output[3];
  func(input, output);
  // CHECK: 2 2 0
  for (int i = 0; i < 3; ++i) {
    printf("%d ", output[i]);
  }
}
#endif