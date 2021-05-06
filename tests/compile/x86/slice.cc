// RUN: %cxx %s -o %t %flags %include %link -DBUILD_IR
// RUN: %t > %t.obj
// RUN: %cxx %s %t.obj -o %t2 %include
// RUN: %t2 2>&1| FileCheck %s

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
      arg_builder.CreateArgument("input", Type{DataType::FLOAT32, {3, 4}});

  BasicBlockBuilder bb_builder(func);
  BasicBlock* bb = bb_builder.CreateBasicBlock("bb0");

  std::vector<int> w0 = {1, 0};
  std::vector<int> w1 = {2, -1};
  ConstantBuilder c_builder(func);
  auto begin =
      c_builder.CreateConstant("begin", Type{DataType::INT32, {2}}, w0.data());
  auto size =
      c_builder.CreateConstant("size", Type{DataType::INT32, {2}}, w1.data());
  IRBuilder ir_builder(bb);

  Instruction* inst = ir_builder.CreateSlice("slice", {*input, *begin, *size});
  ir_builder.CreateReturn("ret", *inst);

  // simulate the driver's argv[0] by reading from env var.
  ctx.SetBasePath(getenv("HALO_BASE_PATH"));

  PassManager pm(ctx);
  pm.AddTypeLegalizerPass(true);
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
extern void func(const float* input, float* output);
}

int main() {
  const float input[3 * 4] = {
      -7.170710712671279907e-02, 8.385377004742622375e-04,
      -4.007652401924133301e-02, -3.674444183707237244e-02,
      -1.560225244611501694e-02, -6.233667954802513123e-02,
      -1.997570320963859558e-02, -2.930975146591663361e-02,
      -3.357923123985528946e-03, -3.058028034865856171e-02,
      -6.031786091625690460e-03, -1.607417315244674683e-02};
  float output[2 * 4];
  func(input, output);
  // clang-format off
// CHECK: -0.01560225 -0.06233668 -0.01997570 -0.02930975 -0.00335792 -0.03058028 -0.00603179 -0.01607417
  // clang-format on
  for (int i = 0; i < 8; ++i) {
    printf("%.8f ", output[i]);
  }
}
#endif