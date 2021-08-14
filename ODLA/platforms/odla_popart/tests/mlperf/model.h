//===- Halo Compiler Generated File --------------------------------===//

typedef struct _odla_computation *odla_computation;
extern "C" {
int model_run(int num_inputs, const void *inputs[], int num_outputs,
              void *outputs[], int batch_size);
int model_init();
int model_fini();
int model_helper(odla_computation comp);
};
