#include "ODLA/odla.h"

__attribute__((annotate("halo_build_computation"))) static void build_model(
    odla_computation comp);

extern "C" {
int atoi(const char*);
}

int main(int argc, char** argv) {
  const char* engine_file_name = "a.engine";
  if (argc >= 2) {
    engine_file_name = argv[1];
  }

  int max_batch_size = argc >= 3 ? atoi(argv[2]) : 0;

  odla_device device = nullptr;
  odla_AllocateDevice(nullptr, ODLA_DEVICE_NVIDIA_TENSORRT, 0, &device);

  odla_computation comp = nullptr;

  odla_CreateComputation(&comp);

  bool is_dynamic_batch = max_batch_size > 0;
  if (is_dynamic_batch) {
    int min_batch_size = 1;
    int opt_batch_size = max_batch_size / 2;
    odla_SetComputationItem(comp, ODLA_DYNAMIC_BATCH,
                            (odla_item_value)&is_dynamic_batch);
    odla_SetComputationItem(comp, ODLA_MIN_BATCH_SIZE,
                            (odla_item_value)&min_batch_size);
    odla_SetComputationItem(comp, ODLA_MAX_BATCH_SIZE,
                            (odla_item_value)&max_batch_size);
    odla_SetComputationItem(comp, ODLA_OPT_BATCH_SIZE,
                            (odla_item_value)&opt_batch_size);
  }
  build_model(comp);

  odla_executable exec = nullptr;
  odla_CompileComputation(comp, device, &exec);

  odla_resource_location loc;
  loc.location_type = ODLA_LOCATION_PATH;
  loc.location = engine_file_name;
  odla_StoreExecutable(loc, exec);

  odla_DestroyExecutable(exec);
  odla_DestroyDevice(device);

  return 0;
}
