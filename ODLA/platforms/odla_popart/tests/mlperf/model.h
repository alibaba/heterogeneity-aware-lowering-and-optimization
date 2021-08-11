//===- Halo Compiler Generated File --------------------------------===//

typedef struct _odla_computation *odla_computation;
extern "C" {
void model(const unsigned int indices[3840],
           const unsigned int input_mask[3840],
           const unsigned int positions[3840],
           const unsigned int segments[3840],
           odla_float16 out_Squad_Gemm[3840 * 2]);
void model_init();
void model_fini();
odla_computation model_helper();
};
