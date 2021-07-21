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

  std::vector<int> multiples{2, 2, 3};
  Constant* c_multiples = c_builder.CreateConstant(
      "starts", Type(DataType::INT32, {3}), multiples.data());

  IRBuilder ir_builder(bb);

  Instruction* tile =
      ir_builder.CreateTile("tile", std::vector<Def>{*c_input, *c_multiples});

  ir_builder.CreateReturn("ret", *tile);

  PassManager pm(ctx);
  pm.AddTypeLegalizerPass();
  pm.AddInstSimplifyPass(false, false, false, false, false, false, false);
  pm.AddDCEPass();
  pm.Run(&m);

  m.Dump();

  const std::unique_ptr<Constant>& first_constant = func->Constants().front();

  first_constant->PrintData(
      &std::cout, first_constant->GetResultType().GetTotalNumOfElements(),
      true);

  // clang-format off
  // CHECK: Constant tile_folding([INT32: 4x6x12]) = [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 5, 6, 7, 8, 5, 6, 7, 8, 5, 6, 7, 8, 9, 10, 11, 12, 9, 10, 11, 12, ...]
  // CHECK: BasicBlock: bb0()
  // CHECK-NEXT: Inst: ret([INT32: 4x6x12]) = return(<tile_folding, 0>:[INT32: 4x6x12])
  // CHECK: 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 5, 6, 7, 8, 5, 6, 7, 8, 5, 6, 7, 8, 9, 10, 11, 12, 9, 10, 11, 12, 9, 10, 11, 12, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 5, 6, 7, 8, 5, 6, 7, 8, 5, 6, 7, 8, 9, 10, 11, 12, 9, 10, 11, 12, 9, 10, 11, 12, 13, 14, 15, 16, 13, 14, 15, 16, 13, 14, 15, 16, 17, 18, 19, 20, 17, 18, 19, 20, 17, 18, 19, 20, 21, 22, 23, 24, 21, 22, 23, 24, 21, 22, 23, 24, 13, 14, 15, 16, 13, 14, 15, 16, 13, 14, 15, 16, 17, 18, 19, 20, 17, 18, 19, 20, 17, 18, 19, 20, 21, 22, 23, 24, 21, 22, 23, 24, 21, 22, 23, 24, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 5, 6, 7, 8, 5, 6, 7, 8, 5, 6, 7, 8, 9, 10, 11, 12, 9, 10, 11, 12, 9, 10, 11, 12, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 5, 6, 7, 8, 5, 6, 7, 8, 5, 6, 7, 8, 9, 10, 11, 12, 9, 10, 11, 12, 9, 10, 11, 12, 13, 14, 15, 16, 13, 14, 15, 16, 13, 14, 15, 16, 17, 18, 19, 20, 17, 18, 19, 20, 17, 18, 19, 20, 21, 22, 23, 24, 21, 22, 23, 24, 21, 22, 23, 24, 13, 14, 15, 16, 13, 14, 15, 16, 13, 14, 15, 16, 17, 18, 19, 20, 17, 18, 19, 20, 17, 18, 19, 20, 21, 22, 23, 24, 21, 22, 23, 24, 21, 22, 23, 24
  // clang-format on
}

int main() { build(); }
