
// clang-format off
// RUN: %cxx %s -DIR_TEST -DSRC_FILE=\"%t.cc\" -DBIN_FILE=\"%t.bin\" -o %t %flags %include %link
// RUN: %t 2>&1| FileCheck %s
// RUN: %cxx -c %s -o %t_main.o
// RUN: %cxx -c %t.cc -I%odla_path/include -o %t.o
// RUN: %cxx %t_main.o %t.o %t.bin %odla_link -o %t.exe -lodla_tensorrt
// RUN: %t.exe 10 | FileCheck %s --check-prefix=EXECUTE
// clang-format on

#include <iostream>
#ifdef IR_TEST
#include <fstream>

#include "halo/lib/ir/ir_builder.h"
#include "halo/lib/pass/pass_manager.h"

using namespace halo;

void build() {
  GlobalContext ctx;
  Module m(ctx, "test_module");

  FunctionBuilder func_builder(&m);
  Function* func = func_builder.CreateFunction("func");
  ArgumentBuilder func_arg_builder(func);
  Type ty_data(DataType::FLOAT32, {2, 3});
  Type ty_int_scalar(DataType::INT32, {});
  Argument* input_data = func_arg_builder.CreateArgument("input_data", ty_data);
  Argument* trip = func_arg_builder.CreateArgument("trips", ty_int_scalar);

  // Build main block.
  ConstantBuilder c_builder(func);
  std::vector<int8_t> loop_range{1};
  auto c_loop_range = c_builder.CreateConstant(
      "loop_range", Type(DataType::BOOL, {}), loop_range.data());
  std::vector<float> zeros(6);
  auto init = c_builder.CreateConstant("init", ty_data, zeros.data());

  BasicBlockBuilder bb_builder(func);
  BasicBlock* bb0 = bb_builder.CreateBasicBlock("bb0");
  IRBuilder ir_builder0(bb0);
  auto loop0 = ir_builder0.CreateLoop("loop0", *trip, *c_loop_range, *init);

  // Build Loop Body.
  BasicBlock* loop_body = bb_builder.CreateBasicBlock("loop_body");
  IRBuilder ir_builder1(loop_body);
  ArgumentBuilder arg_builder(loop_body);
  Argument* arg_loop_cnt =
      arg_builder.CreateArgument("arg_loop_cnt", ty_int_scalar);

  Argument* arg_loop_range =
      arg_builder.CreateArgument("arg_loop_range", Type(DataType::BOOL, {1}));
  Argument* arg_loop_input =
      arg_builder.CreateArgument("arg_loop_input", ty_data);
  Instruction* out = ir_builder1.CreateAdd("add", *arg_loop_input, *input_data);
  ir_builder1.CreateReturn("loop_out", *out);
  loop0->SetBody(loop_body);
  loop_body->SetLoopInst(loop0);

  ir_builder0.CreateReturn("ret", std::vector<Def>{*loop0});

  PassManager pm(ctx);
  pm.AddTypeLegalizerPass();
  pm.AddInstSimplifyPass();

  std::ofstream of_code;
  std::ofstream of_constants;

  std::cout << "Outputs" << SRC_FILE << ", " << BIN_FILE << std::endl;
  of_code.open(SRC_FILE, std::ofstream::binary);
  of_constants.open(BIN_FILE, std::ofstream::binary);

  auto cg = pm.AddGenericCXXCodeGenPass(of_code, std::cout);
  pm.AddX86ConstantWriterPass(of_constants);
  pm.Run(&m);
  m.Dump();

  // clang-format off
// CHECK: Module: test_module
// CHECK: Function: func(input_data[FLOAT32: 2x3], trips[INT32: ])
// CHECK: Constant loop_range([BOOL: ]) = [true]
// CHECK: BasicBlock: bb0
// CHECK: Inst: loop0([FLOAT32: 2x3]) = loop(<trips, 0>:[INT32: ], <loop_range, 0>:[BOOL: ], <init, 0>:[FLOAT32: 2x3]) {Attrs: <body: {{.*}}>}
// CHECK: Inst: ret() = return(<loop0, 0>:[FLOAT32: 2x3])
// CHECK: BasicBlock: loop_body
// CHECK: Inst: add([FLOAT32: 2x3]) = add(<arg_loop_input, 0>:[FLOAT32: 2x3], <input_data, 0>:[FLOAT32: 2x3])
  // clang-format on
}

int main() {
  build();
  return 0;
}

#else
// EXECUTE: 100,200,300,400,500,600
#include <array>
extern "C" {
extern void func(const float* input_data, const int* trips, float* output_data);
}
int main(int argc, char** argv) {
  const std::array<float, 6> in{10, 20, 30, 40, 50, 60};
  int trips = 0;
  if (argc > 1) {
    trips = atoi(argv[1]);
  }
  std::array<float, 6> out;
  func(in.data(), &trips, out.data());
  for (auto& x : out) {
    std::cout << x << ",";
  }
  std::cout << std::endl;
}
#endif