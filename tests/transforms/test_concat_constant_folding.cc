// RUN: %cxx %s -o %t %flags %include %link
// RUN: %t 2>&1| FileCheck %s

#include <numeric>

#include "halo/lib/ir/ir_builder.h"
#include "halo/lib/ir/values.h"
#include "halo/lib/pass/pass_manager.h"
#include "halo/lib/transforms/dce.h"
#include "halo/lib/transforms/inst_simplify.h"

using namespace halo;
using namespace std::literals::string_literals;

struct ConstantDescriptor {
  std::string name;
  std::vector<int64_t> dims;

  int64_t GetNumElements() const {
    return std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<>{});
  }
};

struct TestCase {
  std::string name;
  std::vector<ConstantDescriptor> inputs;
  int axis;
};

struct TestSuite {
  std::string name;
  std::vector<TestCase> test_cases;
};

void RunTestSuite(const TestSuite& test_suite) {
  GlobalContext ctx;
  Module m(ctx, test_suite.name);

  FunctionBuilder func_builder(&m);

  for (const TestCase& test_case : test_suite.test_cases) {
    Function* func = func_builder.CreateFunction(test_case.name);
    ConstantBuilder c_builder(func);

    std::vector<Def> params;

    int number = 1;

    for (const ConstantDescriptor& desc : test_case.inputs) {
      std::vector<int> values(desc.GetNumElements());
      std::iota(values.begin(), values.end(), number);
      number += desc.GetNumElements();

      Constant* c_input = c_builder.CreateConstant(
          desc.name, Type(DataType::INT32, desc.dims), values.data());

      params.push_back(*c_input);
    }

    std::ostringstream bb_name;
    bb_name << "concat_" << test_case.inputs.size() << "_inputs_on_axis_"
            << test_case.axis;

    BasicBlockBuilder bb_builder(func);
    BasicBlock* bb = bb_builder.CreateBasicBlock(bb_name.str());
    IRBuilder ir_builder(bb);

    ConcatInst* concat = ir_builder.CreateConcat("concat", params);
    concat->SetAxis(test_case.axis);
    concat->SetN(params.size());
    ir_builder.CreateReturn("ret", *concat);
  }

  PassManager pm(ctx);
  pm.AddTypeLegalizerPass();
  pm.AddInstSimplifyPass();
  pm.AddDCEPass();
  pm.Run(&m);
  m.Dump();
}

int main() {
  RunTestSuite(
      {.name = "test_concat",
       {
           {.name = "axis_0",
            .inputs = {{.name = "x", .dims = {2, 2, 4}},
                       {.name = "y", .dims = {1, 2, 4}}},
            .axis = 0},
           // clang-format off
           // CHECK: Function: axis_0()
           // CHECK-NEXT: Constant concat_folding([INT32: 3x2x4]) = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
           // CHECK-NEXT: BasicBlock: concat_2_inputs_on_axis_0()
           // CHECK-NEXT: Inst: ret() = return(<concat_folding, 0>:[INT32: 3x2x4])
           // clang-format on
           {.name = "axis_1",
            .inputs = {{.name = "x", .dims = {2, 1, 4}},
                       {.name = "y", .dims = {2, 2, 4}}},
            .axis = 1},
           // clang-format off
           // CHECK: Function: axis_1()
           // CHECK-NEXT: Constant concat_folding([INT32: 2x3x4]) = [1, 2, 3, 4, 9, 10, 11, 12, 13, 14, 15, 16, 5, 6, 7, 8, 17, 18, 19, 20, 21, 22, 23, 24]
           // CHECK-NEXT: BasicBlock: concat_2_inputs_on_axis_1()
           // CHECK-NEXT: Inst: ret() = return(<concat_folding, 0>:[INT32: 2x3x4])
           // clang-format on
           {.name = "axis_2",
            .inputs = {{.name = "x", .dims = {2, 2, 2}},
                       {.name = "y", .dims = {2, 2, 3}}},
            .axis = 2}
           // clang-format off
           // CHECK: Function: axis_2()
           // CHECK-NEXT: Constant concat_folding([INT32: 2x2x5]) = [1, 2, 9, 10, 11, 3, 4, 12, 13, 14, 5, 6, 15, 16, 17, 7, 8, 18, 19, 20]
           // CHECK-NEXT: BasicBlock: concat_2_inputs_on_axis_2()
           // CHECK-NEXT: Inst: ret() = return(<concat_folding, 0>:[INT32: 2x2x5])
           // clang-format on
       }});
}
