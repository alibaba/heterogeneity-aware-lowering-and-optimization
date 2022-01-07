// RUN: %cxx %s -o %t %flags %include %link
// RUN: %t 2>&1| FileCheck %s

#include <numeric>

#include "halo/lib/ir/ir_builder.h"
#include "halo/lib/ir/values.h"
#include "halo/lib/pass/pass_manager.h"
#include "halo/lib/transforms/dce.h"
#include "halo/lib/transforms/inst_simplify.h"

using namespace halo;

void build() {
  GlobalContext ctx;
  Module m(ctx, "test_module");

  FunctionBuilder func_builder(&m);

  Function* func = func_builder.CreateFunction("func");

  BasicBlockBuilder bb_builder(func);
  BasicBlock* bb = bb_builder.CreateBasicBlock("bb0");

  ConstantBuilder c_builder(func);

  std::vector<int> input(2 * 3 * 4);
  std::iota(input.begin(), input.end(), 1);
  Constant* c_input = c_builder.CreateConstant(
      "input", Type(DataType::INT32, {2, 3, 4}), input.data());

  std::vector<int> starts{1};
  Constant* c_starts = c_builder.CreateConstant(
      "starts", Type(DataType::INT32, {1}), starts.data());

  std::vector<int> sizes{2};
  Constant* c_sizes = c_builder.CreateConstant(
      "sizes", Type(DataType::INT32, {1}), sizes.data());

  std::vector<int> steps{1};
  Constant* c_steps = c_builder.CreateConstant(
      "steps", Type(DataType::INT32, {1}), steps.data());

  std::vector<int> axes{1};
  Constant* c_axes =
      c_builder.CreateConstant("axes", Type(DataType::INT32, {1}), axes.data());

  IRBuilder ir_builder(bb);

  Instruction* slice = ir_builder.CreateSlice(
      "slice",
      std::vector<Def>{*c_input, *c_starts, *c_sizes, *c_steps, *c_axes});

  ir_builder.CreateReturn("ret", *slice);

  PassManager pm(ctx);
  pm.AddTypeLegalizerPass();
  pm.AddInstSimplifyPass();
  pm.AddDCEPass();
  pm.Run(&m);

  m.Dump();

  // clang-format off
  // CHECK: Constant slice_folded([INT32: 2x2x4]) = [5, 6, 7, 8, 9, 10, 11, 12, 17, 18, 19, 20, 21, 22, 23, 24]
  // CHECK: BasicBlock: bb0()
  // CHECK-NEXT: Inst: ret([INT32: 2x2x4]) = return(<slice_folded, 0>:[INT32: 2x2x4])
  // clang-format on
}

int main() { build(); }
