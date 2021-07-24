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

  Type ty(DataType::FLOAT32, {1, 2, 2, 3});

  ArgumentBuilder arg_builder(func);
  auto input = arg_builder.CreateArgument("input", ty);

  BasicBlockBuilder bb_builder(func);
  BasicBlock* bb = bb_builder.CreateBasicBlock("bb0");

  IRBuilder ir_builder(bb);

  PoolingMaxInst* poolingmax0 =
      ir_builder.CreatePoolingMax("poolingmax0", *input);
  poolingmax0->SetKsize({1, 2, 2, 1});
  ir_builder.CreateReturn("ret", *poolingmax0);

  // simulate the driver's argv[0] by reading from env var.
  ctx.SetBasePath(getenv("HALO_BASE_PATH"));

  PassManager pm(ctx);
  pm.AddTypeLegalizerPass();
  pm.AddGenericLLVMIRCodeGenPass();
  pm.AddGenericLLVMIRWriterPass(std::cout, false);

  pm.Run(&m);
  // clang-format off
  // CHECK: define void @func(<12 x float>* readonly %input, <3 x float>* %out_poolingmax0) {{.*}} {
  // CHECK-NEXT: bb0:
  // CHECK-NEXT: %0 = bitcast <12 x float>* %input to float*
  // CHECK-NEXT: %poolingmax0 = alloca <3 x float>
  // CHECK-NEXT: %1 = bitcast <3 x float>* %poolingmax0 to float*
  // CHECK-NEXT: call void @_sn_rt_poolingmax_f32_nhwc(float* %1, float* %0, {{.*}})
  // CHECK-NEXT: %2 = bitcast <3 x float>* %out_poolingmax0 to i8*
  // CHECK-NEXT: %3 = bitcast <3 x float>* %poolingmax0 to i8*
  // CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %2, i8* align 4 %3, i64 12, i1 false)
  // CHECK-NEXT: ret void
  // clang-format on
}

int main() { build(); }