#include <fstream>
#include <iostream>

#include "stdlib.h"
#include "string.h"
#include "vodh_common.h"

void* vodh_init(void) {
  void* g_vodh = malloc(2);
  return g_vodh;
}

void vodh_deinit(void* vodh_handle) { free(vodh_handle); }

vodh_ret vodh_get_total_cap(void* vodh_handle, struct vodh_total_cap* cap) {
  cap->vxpu_num = 1;
  cap->support = TYPE_GPU;
  cap->net_bw = 1000;
  cap->net_delay = 10;
  cap->memory = 1024;
  cap->compute = 1024;

  return 0;
}

vodh_ret vodh_get_all_dev_info(void* vodh_handle,
                               struct vodh_dev* allvodh_dev) {
  strcpy(allvodh_dev[0].name, "T4");

  return 0;
}

vodh_ret vodh_get_one_dev_cap(void* vodh_handle, struct vodh_dev* dev,
                              struct vodh_dev_cap* cap) {
  cap->type = TYPE_GPU;
  cap->memory = 1024;

  return 0;
}

vodh_ret vodh_dev_open(void* vodh_handle, struct vodh_dev* dev) { return 0; }

vodh_ret vodh_infer(void* vodh_handle, struct vodh_dev* dev,
                    struct vodh_infer_options* options,
                    struct vodh_infer_result* result) {
  std::cout << "cc file path: " << options->model.model_file << "\n";
  std::cout << "bin file path: " << options->model.weight_file << "\n";

  std::ofstream fcc("/tmp/model.cc", std::ios::out);
  fcc.write(static_cast<char*>(options->model.model_data),
            options->model.model_size);
  fcc.close();
  std::ofstream fwt("/tmp/model.bin", std::ios::binary | std::ios::out);
  fwt.write(static_cast<char*>(options->model.weight_data),
            options->model.weight_size);
  fwt.close();

  std::ofstream fin("/tmp/input.txt", std::ios::out);
  fin << "static const float test_input[1 * 224 * 224 * 3] = {\n"
      << "    ";
  float* ip = reinterpret_cast<float*>(options->input[0]->data);
  for (u32 i = 0; i < (options->input[0]->size) / sizeof(float); i++) {
    fin << ip[i] << ", ";
    if (i > 0 && (i % 3) == 2) {
      fin << "\n    ";
    }
  }
  fin << "};\n";
  fin.close();

  float* op = reinterpret_cast<float*>(result->output[0]->data);
  for (u32 i = 0; i < result->output[0]->size / sizeof(float); i++) {
    op[i] = test_output_ref[i];
  }

  return 0;
}

vodh_ret vodh_dev_close(void* vodh_handle, struct vodh_dev* dev) { return 0; }

void* vodh_malloc(u32 size) {
  void* ptr = malloc(size);
  return ptr;
}

void vodh_free(void* ptr) { free(ptr); }
