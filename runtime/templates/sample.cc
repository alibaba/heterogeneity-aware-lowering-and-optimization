#include "ODLA/odla.h"

__attribute__((annotate("halo_build_computation")))
odla_computation static build_model();

static odla_context ctx;
static odla_computation comp;
static odla_device dev;

static odla_uint32 nr_inputs;
static odla_uint32 nr_outputs;

static int init_model() {
  if (comp == nullptr) {
    comp = build_model();
  }
  if (ctx == nullptr) {
    odla_CreateContext(&ctx);
  }
  odla_GetNumOfArgsFromComputation(comp, &nr_inputs);
  odla_GetNumOfOutputsFromComputation(comp, &nr_outputs);

  return 0;
}

#ifdef __cplusplus
extern "C" {
#endif
void sample_model_fini() {
  odla_DestroyContext(ctx);
  ctx = nullptr;
  odla_DestroyComputation(comp);
  comp = nullptr;
}

int run_sample_model(const void* const inputs[], void* const outputs[]) {
  init_model();
  for (int idx = 0; idx < nr_inputs; ++idx) {
    odla_value val;
    odla_GetArgFromComputationByIdx(comp, idx, &val);
    odla_BindToArgument(val, inputs[idx], ctx);
  }
  for (int idx = 0; idx < nr_outputs; ++idx) {
    odla_value val;
    odla_GetOutputFromComputationByIdx(comp, idx, &val);
    odla_BindToOutput(val, outputs[idx], ctx);
  }
  odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, dev);
}

#ifdef __cplusplus
}
#endif
