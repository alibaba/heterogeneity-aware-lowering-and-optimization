#include <ODLA/odla.h>
#include <stdio.h>

#include <array>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include "ODLA/odla_common.h"
#include "doctest.h"
#include "odla_popart.h"
#include "popart_config.h"




void set_computationItem(odla_computation comp, int ipu_nums);

odla_status generate_model();
odla_status model_helper();

void test_bind_funciton_multithread(float* in, float* out);
void execute_multithread(odla_computation comp, float* in, float* out);
json default_json2(float amp = 0.6,
        const std::string& sdk_version = popart::core::packageHash(),
        const std::string& version = std::string("1.0.0"), int batches_per_step = 1,
        int ipu_num = 1, bool save_model = false,
        const std::string& save_model_path = std::string("odla_popart_saved.onnx"),
        bool load_onnx = false,
        const std::string& load_onnx_path = std::string("not_set.onnx"),
        const std::string& execution_mode = std::string("sequence"),
        const std::string& queue_type = std::string("LockFreeQueue"),
        int queue_capacity = 1024 * 1024, bool debug = false);
json default_json(float amp = 0.6,
        const std::string& sdk_version = popart::core::packageHash(),
        const std::string& version = std::string("1.0.0"), int batches_per_step = 1,
        int ipu_num = 1, bool save_model = false,
        const std::string& save_model_path = std::string("odla_popart_saved.onnx"),
        bool load_onnx = false,
        const std::string& load_onnx_path = std::string("not_set.onnx"),
        const std::string& execution_mode = std::string("sequence"),
        const std::string& queue_type = std::string("LockFreeQueue"),
        int queue_capacity = 1024 * 1024, bool debug = false);
