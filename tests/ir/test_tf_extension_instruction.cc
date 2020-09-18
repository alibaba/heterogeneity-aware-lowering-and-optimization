// RUN: %cxx %s -o %t %flags %include %link
// RUN: %t 2>&1| FileCheck %s

#include "halo/lib/ir/ir_builder.h"
#include "halo/lib/ir/values.h"

using namespace halo;

void build() {
  GlobalContext ctx;
  Module m(ctx, "test_module");

  FunctionBuilder func_builder(&m);

  Function* func = func_builder.CreateFunction("func");
  BasicBlockBuilder bb_builder(func);
  BasicBlock* bb = bb_builder.CreateBasicBlock("bb0");

  std::vector<float> w0{1.0, 2.0, 3.0};

  ConstantBuilder c_builder(func);
  auto x0 = c_builder.CreateConstant("input", Type(DataType::FLOAT32, {3}),
                                     w0.data());

  IRBuilder ir_builder(bb);
  // create from tf.broadcast.
  Instruction* inst0 =
      ir_builder.CreateTFExtension("ext0", {*x0}, 1, "Broadcast");
  // creat an integer list as an attribute
  auto attr0 = Attribute::CreateIntegerList("new_shape", {4, 5, 6, 7});
  inst0->AddOneAttribute(std::move(attr0));

  Instruction* inst1 = ir_builder.CreateTFExtension("ext1", {*x0}, 1, "dummy");

  m.Dump();

  // CHECK: Module: test_module
  // CHECK: Function: func()
  // CHECK: Constant input([FLOAT32: 3]) = [1, 2, 3]
  // CHECK: BasicBlock: bb0
  // clang-format off
  // CHECK-NEXT: Inst: ext0({{.*}}) = tf_Broadcast(<input, 0>:[FLOAT32: 3]) {Attrs: <new_shape: [4, 5, 6, 7]>}
  // clang-format on
  // CHECK-NEXT: Inst: ext1({{.*}}) = !tf_dummy(<input, 0>:[FLOAT32: 3])
}

int main() { build(); }