#include <ODLA/odla.h>
#include <stdio.h>

#include <array>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include "odla_popart.h"
#include "popart_config.h"
#include "ODLA/odla_common.h"




odla_status build_default_model();

void set_computationItem(odla_computation comp, bool is_use_cpu = true, int ipu_nums = 1, int batches_per_step = 1,
 bool enable_engine = false, std::string cache_dir = "/tmp/");

void test_bind_funciton_multithread(float* in, float* out);
void execute_multithread(odla_computation comp, float* in, float* out);

json default_json();