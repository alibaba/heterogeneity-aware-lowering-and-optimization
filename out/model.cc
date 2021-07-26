//===- Halo Compiler Generated File --------------------------------===//

#include <ODLA/odla.h>

extern const float Variable[784 * 10];
extern const float Variable_1_broadcasted_7[1 * 10];
extern "C" {
void mnist_simple(const float x[1 * 784], float out_y[1 * 10]);
void mnist_simple_init();
void mnist_simple_fini();
odla_computation mnist_simple_helper();
};
static odla_computation Comp;
odla_computation mnist_simple_helper() {
  odla_computation comp;
  odla_CreateComputation(&comp);
  auto x = odla_CreateArgument({ODLA_FLOAT32, {.size = 2, .dims = {1, 784}}},
                               (const odla_value_id)("x"));
  auto Variable_ =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 2, .dims = {784, 10}}},
                          Variable, (const odla_value_id) "Variable_");
  auto Variable_1_broadcasted_7_ = odla_CreateConstant(
      {ODLA_FLOAT32, {.size = 2, .dims = {1, 10}}}, Variable_1_broadcasted_7,
      (const odla_value_id) "Variable_1_broadcasted_7_");
  auto MatMul =
      odla_Gemm(x, 0, Variable_, 0, 1, 0, nullptr, {.size = 2, .dims = {1, 10}},
                (const odla_value_id) "MatMul");
  auto add =
      odla_Add(MatMul, Variable_1_broadcasted_7_, (const odla_value_id) "add");
  auto y = odla_Softmax(add, -1, (const odla_value_id) "y");
  odla_SetValueAsOutput(y);
  return comp;
}
void mnist_simple_fini() { odla_DestroyComputation(Comp); }
void mnist_simple_init() {
  if (Comp == nullptr) {
    Comp = mnist_simple_helper();
  }
}

#define SINGLE_THREAD_NO
void mnist_simple(const float x[1 * 784], float out_y[1 * 10]) {
  mnist_simple_init();
#ifdef SINGLE_THREAD
  static odla_context Ctx;
  if (Ctx == nullptr) {
    odla_CreateContext(&Ctx);
  };
#else
  //thread_local odla_context Ctx;
  odla_context Ctx = nullptr;
  odla_CreateContext(&Ctx);
#endif
  odla_BindToArgumentById((const odla_value_id) "x", x, Ctx);
  odla_BindToOutputById((const odla_value_id) "y", out_y, Ctx);
  odla_ExecuteComputation(Comp, Ctx, ODLA_COMPUTE_INFERENCE, nullptr);
}
