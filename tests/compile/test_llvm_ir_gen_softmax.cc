// RUN: %cxx %s -o %t %flags %include %link
// RUN: %t 2>&1| FileCheck %s

#include "halo/lib/ir/ir_builder.h"
#include "halo/lib/ir/values.h"
#include "halo/lib/pass/pass_manager.h"
#include "halo/lib/target/generic_llvmir/generic_llvmir_codegen.h"
#include "halo/lib/transforms/type_legalizer.h"

using namespace halo;

void build() {
  GlobalContext ctx;
  Module m(ctx, "test_module");

  FunctionBuilder func_builder(&m);

  Function* func = func_builder.CreateFunction("func");

  Type ty(DataType::FLOAT32, {2, 4});

  ArgumentBuilder arg_builder(func);
  auto input = arg_builder.CreateArgument("input", ty);

  BasicBlockBuilder bb_builder(func);
  BasicBlock* bb = bb_builder.CreateBasicBlock("bb0");

  IRBuilder ir_builder(bb);

  SoftmaxInst* softmax0 = ir_builder.CreateSoftmax("softmax0", *input);
  softmax0->SetAxis(-1);
  ir_builder.CreateReturn("ret", *softmax0);

  // simulate the driver's argv[0] by reading from env var.
  ctx.SetBasePath(getenv("HALO_BASE_PATH"));

  PassManager pm(ctx);
  pm.AddPass<TypeLegalizer>();
  pm.AddPass<GenericLLVMIRCodeGen>();
  pm.AddPass<GenericLLVMIRWriter>(std::ref(std::cout), false);

  pm.Run(&m);
  // clang-format off
  // CHECK: @input_shape = internal global [2 x i64] [i64 2, i64 4]
  // CHECK: define void @func(<8 x float>* readonly %input, <8 x float>* %out_softmax0) {{.*}} {
  // CHECK-NEXT: bb0:
  // CHECK-NEXT: %0 = bitcast <8 x float>* %input to float*
  // CHECK-NEXT: %softmax0 = alloca <8 x float>
  // CHECK-NEXT: %1 = bitcast <8 x float>* %softmax0 to float*
  // CHECK-NEXT: call void @_sn_rt_softmax_f32(float* %1, float* %0, i64* getelementptr inbounds ([2 x i64], [2 x i64]* @input_shape, i32 0, i32 0), i32 1, i32 2, i64 8)
  // CHECK-NEXT: %2 = bitcast <8 x float>* %out_softmax0 to i8*
  // CHECK-NEXT: %3 = bitcast <8 x float>* %softmax0 to i8*
  // CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %2, i8* align 4 %3, i64 32, i1 false)
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
  // clang-format on
}

int main() { build(); }