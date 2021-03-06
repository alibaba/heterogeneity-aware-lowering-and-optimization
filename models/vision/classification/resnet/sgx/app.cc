
#define TEST_SET 0
//#define TEST_BN_0
#include <vector>

#include "resnet_data.h"
#include "test_util.h"
#ifndef COMPARE_ERROR
#define COMPARE_ERROR 1e-3
#endif

#ifdef NOSGX
#include "model.h"
#else
// === SGX
#include <sgx_defs.h>
#include <sgx_eid.h>
#include <sgx_error.h>
#include <sgx_urts.h>

#include "resnet50_v2_u.h"
// ===
#endif

static float out[1000];
int main(int argc, char** argv) {
  int iter = 100;

#ifdef NOSGX
  resnet50_v2(test_input, out);
#else
  // === SGX
  sgx_enclave_id_t eid;
  auto ret = sgx_create_enclave("model.signed.so", SGX_DEBUG_FLAG, nullptr,
                                nullptr, &eid, nullptr);
  if (ret != SGX_SUCCESS) {
    printf("Failed to create enclave: 0x%x\n", ret);
    return 1;
  } else {
    printf("Enclave created\n");
  }
  // === SGX

  ret = resnet50_v2(eid, const_cast<float*>(test_input), out);
  if (ret != SGX_SUCCESS) {
    printf("Failed to call model: 0x%x\n", ret);
    return 1;
  } else {
    printf("Finished Resnet50 Inference\n");
  }

  resnet50_v2_fini(eid);

  ret = sgx_destroy_enclave(eid);
  if (ret != SGX_SUCCESS) {
    printf("Failed to destroy enclave: 0x%x\n", ret);
    return 1;
  }
#endif

  if (Verify(out, test_output_ref, sizeof(out) / sizeof(out[0]),
             COMPARE_ERROR)) {
    std::cout << "Result verified\n";
    auto begin_time = Now();
    for (int i = 0; i < iter; ++i) {
#ifdef NOSGX
      resnet50_v2(test_input, out);
#else
      ret = resnet50_v2(eid, const_cast<float*>(test_input), out);
#endif
    }
    auto end_time = Now();
    if (iter)
      std::cout << "Inference time: "
                << GetDuration(begin_time, end_time) * 1000 / iter << " ms\n";
    return 0;
  }
  std::cout << " Failed\n";
  return 1;
}