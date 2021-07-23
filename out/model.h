//===- Halo Compiler Generated File --------------------------------===//

typedef struct _odla_computation *odla_computation;
extern "C" {
void mnist_simple(const float x[1 * 784], float out_y[1 * 10]);
void mnist_simple_init();
void mnist_simple_fini();
odla_computation mnist_simple_helper();
};
