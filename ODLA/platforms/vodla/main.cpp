#include <ODLA/odla.h>
#include <iostream>

#include "resnet_data.h"

#define BATCH 1
#define EPSILON 0.00001

extern "C" void model_data(odla_context ctx, unsigned int* ipSize,
                           unsigned int* opSize, int batchSize);
extern "C" odla_computation model_helper(const char* ccFile,
                                         const char* binFile);

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cout << "cc file and bin file path required!\n";
    return 1;
  }

  odla_status ost;
  odla_device device;
  ost = odla_AllocateDevice(NULL, ODLA_DEVICE_DEFAULT, &device);
  if (ost == ODLA_FAILURE) {
    std::cout << "Failed create ODLA device!\n";
    return 1;
  }

  odla_computation ocom;
  ocom = model_helper(argv[1], argv[2]);

  odla_context octx;
  odla_CreateContext(&octx);

  odla_value oval;
  odla_GetArgFromComputationByIdx(ocom, 0, &oval);
  odla_BindToArgument(oval, test_input, octx);
  odla_GetOutputFromComputationByIdx(ocom, 0, &oval);
  void* op = malloc(1001 * sizeof(float));
  odla_BindToOutput(oval, op, octx);

  unsigned int ipSize[] = {sizeof(test_input)};
  unsigned int opSize[] = {1001 * sizeof(float)};
  model_data(octx, ipSize, opSize, BATCH);

  ost = odla_ExecuteComputation(ocom, octx, ODLA_COMPUTE_INFERENCE, device);
  if (ost == ODLA_FAILURE) {
    std::cout << "Remote inference failed!\n";
    return 1;
  }

#ifdef DEBUG
  // compare to reference
  unsigned int opNum = 1;
  for (unsigned int lp = 0; lp < BATCH; lp++) {
    for (unsigned int opIdx = 0; opIdx < opNum; opIdx++) {
      for (unsigned int i = 0; i < opSize[opIdx] / sizeof(float); i++) {
        unsigned int idx = lp * opNum + i;
        float* out = reinterpret_cast<float*>(op);
        if (abs(out[idx] - test_output_ref[i]) > EPSILON) {
          std::cout << "out: " << out[idx] << ", ref: " << test_output_ref[i]
                    << "\n";
          std::cout << "Inference output mismatch @out[" << i << "]\n";
        }
        // std::cout << fref[i] << "\n";
      }
    }
  }
#endif

  odla_DestroyContext(octx);
  odla_DestroyComputation(ocom);
  odla_DestroyDevice(device);

  return 0;
}
