
// REQUIRES: odla_dnnl

// clang-format off
// RUN: %cxx -g -DCG_TEST %s -o %t.gen %flags %include %link -DOUTPUT=%t.cc
// RUN: %t.gen
// RUN: %cxx -O2 %t.cc %t.cc.bin %odla_link %s -I%odla_path/include -o %t.dnnl.exe -lodla_dnnl
// RUN: %t.dnnl.exe 2>&1| FileCheck %s

// clang-format on

#ifdef CG_TEST
#include "halo/lib/ir/ir_builder.h"
#include "halo/lib/ir/values.h"
#include "halo/lib/pass/pass_manager.h"
#include "halo/lib/target/generic_cxx/generic_cxx_codegen.h"
#include "halo/lib/target/cpu/x86/binary/x86_llvmir_codegen.h"
#include "halo/lib/transforms/dce.h"
#include "halo/lib/transforms/inst_simplify.h"
#include "halo/lib/transforms/tfextension_legalizer.h"
#include "halo/lib/transforms/type_legalizer.h"
#include <fstream>

#define str(s) std::string(#s)
#define xstr(s) str(s)

using namespace halo;

static void build() {
  GlobalContext ctx;
  Module m(ctx, "test_module");

  FunctionBuilder func_builder(&m);
  Function* func = func_builder.CreateFunction("func");
  ArgumentBuilder arg_builder(func);
  BasicBlockBuilder bb_builder(func);
  BasicBlock* bb = bb_builder.CreateBasicBlock("bb0");
  constexpr int out_ch = 5;
  constexpr int in_ch = 3;
  constexpr int k_h = 2;
  constexpr int k_w = 2;
  std::vector<float> kernel(out_ch * in_ch * k_h * k_w);
  for (int i = 0, e = kernel.size(); i < e; ++i) {
    kernel[i] = i;
  }

  std::vector<float> scale{0, 1, 2};
  std::vector<float> bias{-10, -9, -8};

  ConstantBuilder c_builder(func);
  auto in = arg_builder.CreateArgument(
      "input", Type(DataType::FLOAT32, {1, in_ch, 5, 5}));

  auto k = c_builder.CreateConstant(
      "kernel", Type(DataType::FLOAT32, {out_ch, in_ch, k_h, k_w}),
      kernel.data());

  auto s = c_builder.CreateConstant(
      "scale", Type(DataType::FLOAT32, {in_ch, 1, 1}), scale.data());

  auto b = c_builder.CreateConstant(
      "bias", Type(DataType::FLOAT32, {in_ch, 1, 1}), bias.data());

  IRBuilder ir_builder(bb);

  Instruction* inst0 = ir_builder.CreateMul("mul", {
                                                       *in,
                                                       *s,
                                                   });

  Instruction* inst1 = ir_builder.CreateAdd("add", {*inst0, *b});

  Conv2DInst* inst2 = ir_builder.CreateConv2D("conv", {*inst1, *k});
  inst2->SetDataFormat(DataFormat::NCHW);
  inst2->SetFilterFormat(DataFormat::NCHW);

  ir_builder.CreateReturn("ret", *inst2);

  std::ofstream of_code;
  std::ofstream of_constants;

  of_code.open(xstr(OUTPUT), std::ofstream::binary);
  of_constants.open(xstr(OUTPUT) + ".bin", std::ofstream::binary);

  PassManager pm(ctx);
  pm.AddPass<TypeLegalizer>(true);
  pm.AddPass<InstSimplify>(false, true, false, false, false, true);
  pm.AddPass<DCE>();
  auto cg =
      pm.AddPass<GenericCXXCodeGen>(std::ref(of_code), std::ref(std::cout));
  pm.AddPass<X86ConstantWriter>(std::ref(of_constants));
  pm.Run(&m);
}

int main() { build(); }

#else

#include <iostream>

extern "C" {
extern void func(const float* in, float* out);
}

int main() {
  float out[5 * 4 * 4];
  float in[1 * 3 * 5 * 5];
  for (size_t i = 0; i < sizeof(in) / sizeof(in[0]); ++i) {
    in[i] = i;
  }
  func(in, out);
  int i = 0;
  for (const float& x : out) {
    std::cout << x << " ";
    ++i;
    if (i % 4 == 0) {
      std::cout << std::endl;
    }
  }
  std::cout << std::endl;
}

// CHECK: 4115  4213  4311  4409
// CHECK: 4605  4703  4801  4899
// CHECK: 5095  5193  5291  5389
// CHECK: 5585  5683  5781  5879

// CHECK: 9251  9493  9735  9977
// CHECK: 10461 10703 10945 11187
// CHECK: 11671 11913 12155 12397
// CHECK: 12881 13123 13365 13607

// CHECK: 14387 14773 15159 15545
// CHECK: 16317 16703 17089 17475
// CHECK: 18247 18633 19019 19405
// CHECK: 20177 20563 20949 21335

// CHECK: 19523 20053 20583 21113
// CHECK: 22173 22703 23233 23763
// CHECK: 24823 25353 25883 26413
// CHECK: 27473 28003 28533 29063

// CHECK: 24659 25333 26007 26681
// CHECK: 28029 28703 29377 30051
// CHECK: 31399 32073 32747 33421
// CHECK: 34769 35443 36117 36791
#endif
