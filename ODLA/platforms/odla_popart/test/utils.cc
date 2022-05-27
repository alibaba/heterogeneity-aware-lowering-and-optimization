#include "utils.hpp"


void set_computationItem(odla_computation comp, int ipu_nums) {
  bool use_ipu_model = 0;
  int ipu_num = ipu_nums;
  int batches_per_step = 1;
  odla_SetComputationItem(comp, ODLA_USE_SIM_MODE,
                          (odla_item_value)&use_ipu_model);
  odla_SetComputationItem(comp, ODLA_PROCESSOR_NUM, (odla_item_value)&ipu_num);
  odla_SetComputationItem(comp, ODLA_BATCHES_PER_STEP,
                          (odla_item_value)&batches_per_step);
}

odla_status generate_model() 
{
  std::vector<float> py = {2};
  std::vector<float> pz = {3};
  auto Input =
    odla_CreateArgument({ODLA_FLOAT32, {.size = 1, .dims = {1}}},
                        (const odla_value_id)("Input"));
  auto py_ = odla_CreateConstant({ODLA_FLOAT32, {.size = 1, .dims = {1}}},
                                py.data(), (const odla_value_id) "Mul_const");
  auto pz_ = odla_CreateConstant({ODLA_FLOAT32, {.size = 1, .dims = {1}}},
                                pz.data(), (const odla_value_id) "Add_const");
  auto Mul = odla_Mul(py_, Input, (const odla_value_id) "Mul");
  auto Sub = odla_Sub(pz_, Mul, (const odla_value_id) "Sub");
  odla_SetValueAsOutput(Sub);
}

odla_status model_helper() 
{
  std::vector<float> py = {2};
  std::vector<float> pz = {3};
  auto Input =
    odla_CreateArgument({ODLA_FLOAT32, {.size = 1, .dims = {1}}},
                        (const odla_value_id)("Input"));
  auto py_ = odla_CreateConstant({ODLA_FLOAT32, {.size = 1, .dims = {1}}},
                                py.data(), (const odla_value_id) "Mul_const");
  auto pz_ = odla_CreateConstant({ODLA_FLOAT32, {.size = 1, .dims = {1}}},
                                pz.data(), (const odla_value_id) "Add_const");
  auto Mul = odla_Mul(py_, Input, (const odla_value_id) "Mul");
  auto Add = odla_Add(pz_, Mul, (const odla_value_id) "Add");
  odla_SetValueAsOutput(Add);
}

void test_bind_funciton_multithread(float* in, float* out) 
{
  odla_context ctx_multithread;
  CHECK_EQ(odla_CreateContext(&ctx_multithread), ODLA_SUCCESS);
  CHECK_EQ(odla_BindToArgumentById((const odla_value_id)"Input", in, ctx_multithread), ODLA_SUCCESS);
  CHECK_EQ(odla_BindToOutputById((const odla_value_id) "Add", out, ctx_multithread), ODLA_SUCCESS);
  odla_DestroyContext(ctx_multithread);

}

void execute_multithread(odla_computation comp, float* in, float* out)
{
    odla_context ctx_multithread;
    odla_CreateContext(&ctx_multithread);

    odla_BindToArgumentById((const odla_value_id) "Input", in, ctx_multithread);
    odla_BindToOutputById((const odla_value_id) "Add", out, ctx_multithread);
    odla_ExecuteComputation(comp, ctx_multithread, ODLA_COMPUTE_INFERENCE, nullptr);

    odla_DestroyContext(ctx_multithread);
}

 json default_json2(
        float amp,
        const std::string& sdk_version,
        const std::string& version, int batches_per_step,
        int ipu_num, bool save_model,
        const std::string& save_model_path,
        bool load_onnx,
        const std::string& load_onnx_path,
        const std::string& execution_mode,
        const std::string& queue_type,
        int queue_capacity, bool debug) {
      // Create a json object & fill with default value
      json jsonfile;
      jsonfile["amp"] = amp;
      jsonfile["sdk_version"] = sdk_version;
      jsonfile["version"] = version;
      jsonfile["batches_per_step"] = batches_per_step;
      jsonfile["ipu_num"] = ipu_num;
      jsonfile["save_model"] = save_model;
      jsonfile["save_model_path"] = save_model_path;
      jsonfile["load_onnx"] = false;
      jsonfile["load_onnx_path"] = load_onnx_path;
      jsonfile["execution_mode"] = execution_mode;
      jsonfile["queue_type"] = queue_type;
      jsonfile["queue_capacity"] = queue_capacity;
      jsonfile["debug"] = debug;

      json pipeline;
      std::vector<int> vec1, vec2;
      vec1.push_back(0);
      vec1.push_back(0);
      vec2.push_back(1);
      vec2.push_back(1);
      pipeline["Input"] = vec1;
      pipeline["Mul"] = vec2;
      pipeline["Mul_const"] = vec2;
      pipeline["Add"] = vec2;
      pipeline["Add_const"] = vec2;

      jsonfile["pipeline"] = pipeline;

      std::ofstream file("/tmp/tmp.json");
      file << jsonfile;
      return jsonfile;
    }



 json default_json(
        float amp,
        const std::string& sdk_version,
        const std::string& version, int batches_per_step,
        int ipu_num, bool save_model,
        const std::string& save_model_path,
        bool load_onnx,
        const std::string& load_onnx_path,
        const std::string& execution_mode,
        const std::string& queue_type,
        int queue_capacity, bool debug) {
      // Create a json object & fill with default value
      json jsonfile;
      jsonfile["amp"] = amp;
      jsonfile["sdk_version"] = sdk_version;
      jsonfile["version"] = version;
      jsonfile["batches_per_step"] = batches_per_step;
      jsonfile["ipu_num"] = ipu_num;
      jsonfile["save_model"] = save_model;
      jsonfile["save_model_path"] = save_model_path;
      jsonfile["load_onnx"] = false;
      jsonfile["load_onnx_path"] = load_onnx_path;
      jsonfile["execution_mode"] = execution_mode;
      jsonfile["queue_type"] = queue_type;
      jsonfile["queue_capacity"] = queue_capacity;
      jsonfile["debug"] = debug;

      json pipeline;
      jsonfile["pipeline"] = pipeline;

      std::ofstream file("/tmp/tmp.json");
      file << jsonfile;
      file.close();

      return jsonfile;
    }

