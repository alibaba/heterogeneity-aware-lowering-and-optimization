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

  ArgumentBuilder arg_builder(func);
  auto input =
      arg_builder.CreateArgument("input", Type{DataType::FLOAT32, {1, 3}});

  BasicBlockBuilder bb_builder(func);
  BasicBlock* bb = bb_builder.CreateBasicBlock("bb0");

  std::vector<float> w0{1.0, 2.0, 3.0};

  ConstantBuilder c_builder(func);
  auto w = c_builder.CreateConstant("w0", Type{DataType::FLOAT32, {3, 1}},
                                    w0.data());

  IRBuilder ir_builder(bb);

  Instruction* mm = ir_builder.CreateMatMul("mm", *input, *w);
  ir_builder.CreateReturn("ret", *mm);

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

  // CHECK: void @func(<3 x float>* readonly %input, <1 x float>* %out_mm)  {{.*}} {
  // CHECK: bb0:
  // CHECK:   %0 = bitcast <3 x float>* %input to float*
  // CHECK:   %mm = alloca <1 x float>
  // CHECK:   %1  = bitcast <1 x float>* %mm to float*
  // CHECK:   call void @_sn_rt_matmul_f32(float* %1, float* %0
  // CHECK:   %2 = bitcast <1 x float>* %out_mm to i8*
  // CHECK:   %3 = bitcast <1 x float>* %mm to i8*
  // CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %2, i8* align 4 %3, i64 4, i1 false)
  // CHECK:   ret void

  // CHECK: define {{.*}} void @_sn_rt_matmul_f32
  // CHECK:   ret void
  // CHECK: }
  // clang-format on
}

int main() { build(); }