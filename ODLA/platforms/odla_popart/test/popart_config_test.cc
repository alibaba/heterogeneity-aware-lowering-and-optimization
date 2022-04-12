#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <ODLA/odla.h>

#include <fstream>
#include <iostream>
#include "popart_config.h"
#include "ODLA/odla_common.h"
#include "doctest.h"
#include "json.hpp"
 using json = nlohmann::json;

json default_json(
    float amp = 0.6,
    const std::string& sdk_version = popart::core::packageHash(),
    const std::string& version = std::string("1.0.0"), int batches_per_step = 1,
    int ipu_num = 1, bool save_model = false,
    const std::string& save_model_path = std::string("odla_popart_saved.onnx"),
    bool load_onnx = false,
    const std::string& load_onnx_path = std::string("not_set.onnx"),
    const std::string& execution_mode = std::string("sequence"),
    const std::string& queue_type = std::string("LockFreeQueue"),
    int queue_capacity = 1024 * 1024, bool debug = false) {
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
  pipeline["^embedding_"] = vec1;
  pipeline["^layer[0-9]_"] = vec1;
  pipeline["^layer[0-1]_"] = vec1;
  pipeline["^layer1[2-9]_"] = vec2;
  pipeline["^layer2[0-3]_"] = vec2;
  pipeline["^squad_"] = vec2;

  jsonfile["pipeline"] = pipeline;


  std::ofstream file("/tmp/tmp.json");
  file << jsonfile;
  return jsonfile;
}

TEST_CASE("testing popart_config") {
  /*SUB_CASE("testing the execution mode PIPELINE_ASYNC") {
    using jsonf = nlohmann::json;
    jsonf jsonfile;
    jsonfile["execution_mode"] = "pipeline_async";

    std::string config_file_path("/tmp/tmp.json");
    std::ofstream file(config_file_path);
    file << jsonfile;
  }
  */
    //Check parse_from_json
    float _amp = 0.6;
    int _batches_per_step = 1;
    int _ipu_num = 1;
    bool _save_model = false;
    const std::string _sdk_version = popart::core::packageHash();
    const std::string _version = std::string("1.0.0");
    const std::string _save_model_path = std::string("odla_popart_saved.onnx");
    bool _load_onnx = false;
    const std::string _load_onnx_path = std::string("not_set.onnx");
    const std::string _execution_mode = std::string("sequence");
    const std::string _queue_type = std::string("LockFreeQueue");
    int _queue_capacity = 1024 * 1024;
    bool _debug = false;
    ExecutionMode _Queue_type = SEQUENCE;

    json _config_json = default_json(_amp, _sdk_version, _version, _batches_per_step, _ipu_num, 
        _save_model, _save_model_path, _load_onnx, _load_onnx_path, _execution_mode, _queue_type, _queue_capacity, _debug);


   PopartConfig::instance()->parse_from_json(_config_json);

   CHECK(PopartConfig::instance()->inited() == true);
   CHECK(PopartConfig::instance()->amp() == _amp);
   CHECK(PopartConfig::instance()->batches_per_step() == _batches_per_step);
   CHECK(PopartConfig::instance()->ipu_num() == _ipu_num);
   CHECK(PopartConfig::instance()->save_model() == _save_model);
   CHECK(PopartConfig::instance()->version() == _version);
   CHECK(PopartConfig::instance()->save_model_path() == _save_model_path);
   CHECK(PopartConfig::instance()->load_onnx() == _load_onnx);
   CHECK(PopartConfig::instance()->load_onnx_path() == _load_onnx_path);
   CHECK(PopartConfig::instance()->execution_mode() == _Queue_type);
   CHECK(PopartConfig::instance()->queue_type() == _queue_type);
   CHECK(PopartConfig::instance()->queue_capacity() == _queue_capacity);
   CHECK(PopartConfig::instance()->debug() == _debug);


   //get_pipeline_setting
   int64_t ipu_idx = -1, pipeline_stage = -1;
   PopartConfig::instance()->get_pipeline_setting("embedding_1", ipu_idx, 
       pipeline_stage);
   CHECK(ipu_idx == 0);
   CHECK(pipeline_stage == 0);

   PopartConfig::instance()->get_pipeline_setting("layer0_", ipu_idx,
       pipeline_stage);
   CHECK(ipu_idx == 0);
   CHECK(pipeline_stage == 0);

   PopartConfig::instance()->get_pipeline_setting("layer12_", ipu_idx,
       pipeline_stage);
   CHECK(ipu_idx == 1);
   CHECK(pipeline_stage == 1);

   PopartConfig::instance()->get_pipeline_setting("layer20_", ipu_idx,
       pipeline_stage);
   CHECK(ipu_idx == 1);
   CHECK(pipeline_stage == 1);



}
