// RUN: %cxx %s -o %t %flags %include %link
// RUN: %t 2>&1| FileCheck %s
// REQUIRES: halo_rtlib

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

  std::vector<float> w0{1.0, 2.0, 3.0};
  std::vector<float> w1{4.0, 5.0, 6.0};

  ConstantBuilder c_builder(func);
  auto c0 = c_builder.CreateConstant("w0", ty, w0.data());
  auto c1 = c_builder.CreateConstant("w1", ty, w1);

  IRBuilder ir_builder(bb);

  Instruction* add0 = ir_builder.CreateAdd("add0", *input, *c0);
  Instruction* add1 = ir_builder.CreateAdd("add1", *add0, *c1);
  ir_builder.CreateReturn("ret", *add1);

  // simulate the driver's argv[0] by reading from env var.
  ctx.SetBasePath(getenv("HALO_BASE_PATH"));

  PassManager pm(ctx);
  pm.AddTypeLegalizerPass();
  pm.AddGenericLLVMIRCodeGenPass();
  pm.AddGenericLLVMIRWriterPass(std::cout, false);

  pm.Run(&m);

  // clang-format off
  // CHECK: ModuleID = 'test_module'
  // CHECK: @w0 = internal global <3 x float> <float 1.000000e+00,
  // CHECK: @w1 = internal global <3 x float> <float 4.000000e+00,
  // CHECK: void @func(<3 x float>* readonly %input, <3 x float>* %out_add1)  {{.*}} {
  // CHECK: bb0:
  // CHECK:   %add0 = alloca <3 x float>
  // CHECK:   %0 = bitcast <3 x float>* %add0 to float*
  // CHECK:   %1 = bitcast <3 x float>* %input to float*
  // CHECK:   call void @_sn_rt_add_f32(float* %0, float* %1, {{.*}}, i64 3, i1 false, i64* null, i64* null, i64* null, i32 1)
  // CHECK:   %add1 = alloca <3 x float>
  // CHECK:   %2 = bitcast <3 x float>* %add1 to float*
  // CHECK:   %3 = bitcast <3 x float>* %add0 to float*
  // CHECK:   call void @_sn_rt_add_f32(float* %2, float* %3, {{.*}}, i64 3, i1 false, i64* null, i64* null, i64* null, i32 1)
  // CHECK:   %4 = bitcast <3 x float>* %out_add1 to i8*
  // CHECK:   %5 = bitcast <3 x float>* %add1 to i8*
  // CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %4, i8* align 4 %5, i64 12, i1 false)
  // CHECK:   ret void
  // CHECK: }
  // clang-format on
}

int main() { build(); }