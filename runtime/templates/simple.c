#include "ODLA/odla.h"

__attribute__((annotate("halo_build_computation")))
odla_computation static build_odla_computation();

static odla_context ctx;
static odla_computation comp;

__attribute__((annotate("halo_nr_inputs"))) static const int nr_inputs = 0;
__attribute__((annotate("halo_nr_outputs"))) static const int nr_outputs = 0;

__attribute__((annotate("halo_input_ids"))) static const char* InputIds[] = {};
__attribute__((
    annotate("halo_output_ids"))) static const char* OutputIds[] = {};
__attribute__((annotate(
    "halo_output_types"))) static const odla_value_type OutputTypes[] = {};

static void BindInputsOutputs(const void* inputs[], void* outputs[]) {
  for (int i = 0; i < nr_inputs; ++i) {
    odla_BindToArgumentById((const odla_value_id)InputIds[i], inputs[i], ctx);
  }
  for (int i = 0; i < nr_outputs; ++i) {
    odla_BindToOutputById((const odla_value_id)OutputIds[i], outputs[i], ctx);
  }
}

int get_num_inputs() { return nr_inputs; }

int get_num_outputs() { return nr_outputs; }

odla_value_type get_output_type(int idx) {
  if (idx >= 0 && idx < nr_outputs) return OutputTypes[idx];
  return (odla_value_type){ODLA_BOOL, (odla_value_shape){-1, {}}};
}

int model_init() {
  if (comp == NULL) {
    comp = build_odla_computation();
  }
  if (ctx == NULL) {
    odla_CreateContext(&ctx);
  }
  return 0;
}

int model_fini() {
  odla_DestroyComputation(comp);
  return 0;
}
int model_run(const void* inputs[], void* outputs[]) {
  BindInputsOutputs(inputs, outputs);
  return odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, NULL);
}
