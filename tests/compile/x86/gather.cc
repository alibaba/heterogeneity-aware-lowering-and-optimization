// RUN: %cxx %s -o %t %flags %include %link -DBUILD_IR
// RUN: %t > %t.obj
// RUN: %cxx %s %t.obj -o %t2 %include
// RUN: %t2 2>&1| FileCheck %s
// REQUIRES: halo_rtlib

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
      arg_builder.CreateArgument("input", Type{DataType::INT32, {1, 4, 1}});

  BasicBlockBuilder bb_builder(func);
  BasicBlock* bb = bb_builder.CreateBasicBlock("bb0");

  // parse from npy
  std::vector<float> w0 = {
      -7.170710712671279907e-02, 8.385377004742622375e-04,
      -4.007652401924133301e-02, -3.674444183707237244e-02,
      -1.560225244611501694e-02, -6.233667954802513123e-02,
      -1.997570320963859558e-02, -2.930975146591663361e-02,
      -3.357923123985528946e-03, -3.058028034865856171e-02,
      -6.031786091625690460e-03, -1.607417315244674683e-02,
      -2.133625559508800507e-02, -5.314207077026367188e-02,
      -6.714689079672098160e-03, -4.212854057550430298e-02,
      -4.732817411422729492e-02, 7.878166623413562775e-03,
      -6.828416138887405396e-02, -3.257048130035400391e-02,
      -4.365085437893867493e-02, -7.706623058766126633e-03,
      -9.249377064406871796e-03, -3.120727837085723877e-02,
      -3.675316274166107178e-02, -3.804220259189605713e-02,
      -1.424242556095123291e-02, -1.985356956720352173e-02,
      -3.720996901392936707e-02, -9.751518256962299347e-03};

  ConstantBuilder c_builder(func);
  auto w = c_builder.CreateConstant("w0", Type{DataType::FLOAT32, {10, 3}},
                                    w0.data());

  IRBuilder ir_builder(bb);

  GatherInst* gather = ir_builder.CreateGather("gather", {*w, *input});
  gather->SetAxis(0);
  ir_builder.CreateReturn("ret", *gather);

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
extern void func(const int* input, float* output);
}

int main() {
  const int input[4] = {7, 2, 2, 3};
  float output[4 * 3];
  func(input, output);
  // clang-format off
// CHECK: -0.00770662 -0.00924938 -0.03120728 -0.01997570 -0.02930975 -0.00335792 -0.01997570 -0.02930975 -0.00335792 -0.03058028 -0.00603179 -0.01607417
  // clang-format on
  for (int i = 0; i < 12; ++i) {
    printf("%.8f ", output[i]);
  }
}
#endif