// RUN: %cxx %s -o %t %flags %include %link
// RUN: %t 2>&1| FileCheck %s

#include "halo/lib/ir/ir_builder.h"
#include "halo/lib/pass/pass_manager.h"
#include "halo/lib/target/generic_cxx/generic_cxx_codegen.h"
#include "halo/lib/transforms/type_legalizer.h"

using namespace halo;

void build(Module* m) {
  FunctionBuilder func_builder(m);

  Type ty{DataType::FLOAT32, std::vector<int64_t>{3}};
  Function* callee = func_builder.CreateFunction("callee");
  callee->SetAsEntryFunction(false);
  {
    ArgumentBuilder arg_builder(callee);
    Argument* arg0 = arg_builder.CreateArgument("arg0", ty);

    Argument* arg1 = arg_builder.CreateArgument("arg1", ty);

    BasicBlockBuilder bb_builder(callee);
    BasicBlock* bb = bb_builder.CreateBasicBlock("bb0");

    std::vector<float> w_v{1.0, 2.0, 3.0};
    ConstantBuilder c_builder(callee);
    auto w0 = c_builder.CreateConstant("w0", ty, w_v.data());
    IRBuilder ir_builder(bb);

    Instruction* mul = ir_builder.CreateMul("mul", *arg0, *w0);
    Instruction* add = ir_builder.CreateAdd("add", *mul, *arg1);

    ir_builder.CreateReturn("ret", std::vector<Def>{*add});
  }

  Function* caller = func_builder.CreateFunction("caller");
  caller->SetAsEntryFunction(true);
  {
    ArgumentBuilder arg_builder(caller);
    Argument* arg0 = arg_builder.CreateArgument("arg0", ty);

    BasicBlockBuilder bb_builder(caller);
    BasicBlock* bb = bb_builder.CreateBasicBlock("bb0");

    std::vector<float> w_v{4.0, 5.0, 6.0};
    ConstantBuilder c_builder(caller);
    auto w0 = c_builder.CreateConstant("w1", ty, w_v.data());
    IRBuilder ir_builder(bb);

    auto call = ir_builder.CreateCall("call", {*arg0, *w0});
    call->SetCallee(callee);
    call->SetNumOfResults(1);

    ir_builder.CreateReturn("ret", {Def{call, 0}});
  }

  // clang-format off
// CHECK: Module: test_module
// CHECK: Function: callee(arg0[FLOAT32: 3], arg1[FLOAT32: 3])
// CHECK: Inst: mul([FLOAT32: 3]) = mul(<arg0, 0>:[FLOAT32: 3], <w0, 0>:[FLOAT32: 3])
// CHECK: Inst: add([FLOAT32: 3]) = add(<mul, 0>:[FLOAT32: 3], <arg1, 0>:[FLOAT32: 3])
// CHECK: Inst: ret() = return(<add, 0>:[FLOAT32: 3])
// CHECK: Function: caller(arg0[FLOAT32: 3])
// CHECK: Constant w1([FLOAT32: 3]) = [4, 5, 6]
// CHECK: Inst: call([FLOAT32: 3]) = call(<arg0, 0>:[FLOAT32: 3], <w1, 0>:[FLOAT32: 3])
// CHECK: Inst: ret() = return(<call, 0>:[FLOAT32: 3])
// clang-format  on
}

int main() {
  GlobalContext ctx;
  Module m(ctx, "test_module");
  build(&m);
  PassManager pm(ctx);
  pm.AddPass<TypeLegalizer>();
  pm.Run(&m);
  m.Dump();
}