First, compile the model into ODLA C++ code:
```bash
halo -target cxx mnist_simple.pb -o out/model.cc

From generated *model.h*:
```c++
extern "C" {
void mnist_simple(const float x[1 * 784], float out_y[1 * 10]);
void mnist_simple_init();
void mnist_simple_fini();
};
```

*mnist_simple()* is the entry function to do the inference, which takes array *x* as input and results will be written into *out_y*.
*mnist_simple()* can be called multiple times while
*mnist_simple_init()* and *minist_simple_fini()* are called once to initialize and to cleanup the whole computation process, respectively.


*model.cc* is an ODLA-based C++ file. The main part of it is to build
the model computation using ODLA APIs:

```c++
// Graph building
static void mnist_simple_helper() {
  odla_CreateComputation(&Comp);
  auto x = odla_CreateArgument({ODLA_FLOAT32, {.size = 2, .dims = {1, 784}}},
                               (const odla_value_id)("x"));
  auto V =
      odla_CreateConstant({ODLA_FLOAT32, {.size = 2, .dims = {784, 10}}},
                          Variable, (const odla_value_id) "V");
  auto V1 = odla_CreateConstant(
      {ODLA_FLOAT32, {.size = 2, .dims = {1, 10}}}, Variable_1_broadcasted_7,
      (const odla_value_id) "V1");
  auto MatMul =
      odla_Gemm(x, 0, V, 0, 1, 0, nullptr, {.size = 2, .dims = {1, 10}},
                (const odla_value_id) "MatMul");
  auto add =
      odla_Add(MatMul, V1, (const odla_value_id) "add");
  auto y = odla_Softmax(add, -1, (const odla_value_id) "y");
  odla_SetValueAsOutput(y);
}

// Entry function
void mnist_simple(const float x[1 * 784], float out_y[1 * 10]) {
  // ...some setup code skipped.
  mnist_simple_init(); // it calls mnist_simple_helper() once.
  odla_BindToArgumentById((const odla_value_id) "x", x, Ctx);
  odla_BindToOutputById((const odla_value_id) "y", out_y, Ctx);
  odla_ExecuteComputation(Comp, Ctx, ODLA_COMPUTE_INFERENCE, nullptr);
}
```

The code snippet of the demo application:

```c++
#include "model.h" // include the generated header.
int main(int argc, char** argv) {
  //... read 1000 images & labels.
  mnist_simple_init(); // Initialize computation.
  
  int correct = 0;
  for (int i = 0; i < 1000; ++i) {
    std::array<float, 28 * 28> input;
    std::array<float, 10> output;
    // ... preprocess inputs
    mnist_simple(input.data(), output.data());
    int pred = std::max_element(output.begin(), output.end()) - output.begin();
    correct += (pred == labels[i]);
  }
  std::cout << "Accuracy: " << << correct / 1000.0 << "% \n";
  mnist_simple_fini(); // Clean up.
}
```

Next, we can use any modern C++ compiler to compile the generated code:

```bash
g++ out/model.cc -I<halo_install_path>/include -c -o out/model.o
g++ main.cc -Iout -c -o out/main.o
```
Assume we link it with a DNNL based ODLA accelerating runtime library:
```bash
g++ -o out/demo out/main.o out/model.o out/model.bin \
  -L<halo_install_path>/lib/ODLA -lodla_dnnl -Wl,-rpath=<halo_install_path>/lib/ODLA
```
To switch to the TensorRT based ODLA runtime, just simply replace "-lodla_dnnl" with "-lodla_tensorrt".


MNIST example code can be found [here](models/vision/classification/mnist_simple)
