#include <fstream>
#include <iostream>
#include <vector>
#include <thread>
#include <iomanip>
#include <ODLA/odla.h>
#include "common.h"
#include "json.hpp"
#include "cnpy.h"

Config* Config::_instance = new Config();

void Config::load(const std::string& file_path)
{
    using json = nlohmann::json;
    std::ifstream ifs(file_path);
    json jf = json::parse(ifs);
    
    std::vector<std::string> required = {"call_method", "dlopen_times", "model",
        "thread_number", "npz", "inputs", "outputs", "result", 
        "thread_buffer_cnt", "duration"
    };
    for(auto &name : required)
        if(!jf.contains(name))
            throw std::invalid_argument("[" + name + "] must be configured in the config file");
    
    _call_method        = jf["call_method"].get<std::string>();
    _dlopen_times       = jf["dlopen_times"].get<std::uint32_t>();
    _model_path         = jf["model"].get<std::string>();
    _thread_number      = jf["thread_number"].get<std::uint32_t>();
    _npz_file_path      = jf["npz"].get<std::string>();
    _inputs             = jf["inputs"].get<std::map<std::string, std::vector<std::string>>>();
    _outputs            = jf["outputs"].get<std::map<std::string, std::vector<std::string>>>();
    _result_file        = jf["result"].get<std::string>();
    _thread_buffer_cnt  = jf["thread_buffer_cnt"].get<std::uint32_t>();
    std::string tmp_du  = jf["duration"].get<std::string>();
    char unit = tmp_du.back();
    tmp_du.pop_back();
    int factor = 1;
    if(unit == 'm' || unit == 'M')
        factor = 60;
    else if(unit == 'h' || unit == 'H')
        factor = 60 * 60;
    else if(unit != 's' && unit != 'S')
        throw std::invalid_argument("duration must be end with [sSmMhH]: " + tmp_du);
    _duration = stol(tmp_du) * factor;
    print();
}

void Config::print()
{
    std::string line(80, '=');
    std::cout << line << std::endl;
    std::cout << "call methoud: " << _call_method << std::endl;
    std::cout << "dlopen_times: " << _dlopen_times << std::endl;
    std::cout << "model path: " << _model_path << std::endl;
    std::cout << "npz file path: " << _npz_file_path << std::endl;
    std::cout << "thread number: " << _thread_number << std::endl;
    std::cout << "thread buffer count: " << _thread_buffer_cnt << std::endl;
    std::cout << "result will be saved to: " << _result_file << std::endl;
    std::cout << "duration: " << _duration << std::endl;
    for(auto& input : _inputs)
        std::cout << input.first << " <-----> [" << input.second[0] 
                  << ", " << input.second[1] << "]" << std::endl;
    for(auto& output : _outputs)
        std::cout << output.first << " <-----> [" << output.second[0] 
                  << ", " << output.second[1] << "]" << std::endl;
    std::cout << line << std::endl;
}

/*---------------------------------------------------------------------------------------*/
void BaseTest::save_latency_results(const std::vector<float>& latencies, 
    const std::map<std::string, std::vector<cnpy::NpyArray>>& results)
{
    std::string result_file = Config::instance()->result_file();
    std::cout << "Write the latencies to file: " << result_file << std::endl;
    cnpy::npz_save<float>(result_file, "latencies", &latencies[0], {latencies.size()}, "w");

    for(auto &result : results){
         bool all_result_equal = true;
        for(int i = 0; i < result.second.size(); i++)
        {
            if(std::memcmp(result.second[0].data<unsigned char>(), 
                result.second[i].data<unsigned char>(), 
                result.second[i].num_bytes()) != 0)
            {
                std::cerr << "The result [" << i << "] is not same with result 0." << std::endl;
                all_result_equal = false;
                break;
            }
        }
        if(all_result_equal)
            std::cout << "All results are equal as expected, will write the first data into the " 
                      << result_file << " file." << std::endl;
        //If the result is float16, need a convert
        auto outputs = Config::instance()->outputs();
        if(outputs[result.first][1] == "FP16")
        {
            std::cout << "Convert the result from odla_float16 to float" << std::endl;
            std::vector<float> float_result;
            std::cout << "The number of values in results[0] is: " << result.second[0].num_vals << std::endl;
            //assert(result[0].num_vals == 384*2*10*128);
            const odla_float16 *data = result.second[0].data<odla_float16>();
            for(int i=0; i<result.second[0].num_vals; i++) 
            {
                float_result.push_back(Float::GetFP32(*(data+i)));
            }
            cnpy::npz_save<float>(result_file, result.first, &float_result[0], {result.second[0].num_vals}, "a");
        }
        else if(outputs[result.first][1] == "UNIT32")
            cnpy::npz_save<std::uint32_t>(result_file, result.first, 
                result.second[0].data<std::uint32_t>(), 
                {result.second[0].num_vals}, "a");
        else
            throw std::invalid_argument("Need to support the type: " + outputs[result.first][1] + "for result save");
        std::cout << "Write the latencies to file: " << result_file << std::endl;
        
    }
}

cnpy::npz_t* BaseTest::prepare_data()
{
  const std::uint32_t data_count = Config::instance()->thread_number() * Config::instance()->thread_buffer_cnt();
  cnpy::npz_t* all_data = new cnpy::npz_t[data_count];
  cnpy::npz_t one_data = cnpy::npz_load(Config::instance()->npz_file_path()); 
  for(int i=0; i < data_count; i++)
    all_data[i] = one_data;
  std::cout << "data prepared" << std::endl;
  return all_data;
}

void inference(int thread_id, cnpy::npz_t* all_data, BaseTest* test){
    std::vector<float> latencies;
    //May be more than one results
    std::vector<std::string> outputs_tensorids;
    std::map<std::string, std::vector<cnpy::NpyArray>> results;
    for(auto &output:Config::instance()->outputs()){
        std::vector<cnpy::NpyArray> new_vec;
        outputs_tensorids.push_back(output.first);
        results[output.first] = new_vec;
    }
  
    auto very_start = std::chrono::steady_clock::now();
    while(true){
        static int idx = 0;
        static int counter = 0;
        cnpy::npz_t& data = *(all_data + idx);
        idx = (idx+1) % Config::instance()->thread_buffer_cnt();  //in case the data was not written before reused.
        auto start = std::chrono::steady_clock::now();

        test->do_inference(data); //should implement this for different caller

        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<float, std::milli> elapsed_milliseconds = end-start;
        auto latency = elapsed_milliseconds.count();
        latencies.push_back(latency);
        for(auto &id : outputs_tensorids)
            results[id].push_back(data[id]);
        std::chrono::duration<float> worked_duration = end - very_start;
        int time_left = Config::instance()->duration() - worked_duration.count();
        if(time_left <= 0){
            std::cout << "Hit the duration: " << Config::instance()->duration() 
                      << " with " << worked_duration.count() << std::endl;
            break;
        }
        else if(time_left / 10 != counter){
            std::cout << "\r[Time left: " << std::setw(20) << time_left << "]" << std::flush;
            counter = time_left / 10;
        }
    }
    std::cout << "==================== Inference loop finished ==============================" << std::endl;
    test->save_latency_results(latencies, results);
}

void BaseTest::start(const std::string& config_file)
{
    Config::instance()->load(config_file);
    assert(Config::instance()->thread_buffer_cnt() > 2);  //Ensure more than 2 data can compare
    cnpy::npz_t* all_data = prepare_data();

    prerequisites();

    auto start = std::chrono::steady_clock::now();
    if(Config::instance()->thread_number() == 1){
        inference(0, all_data, this);
    }
    else{
        std::thread threads[Config::instance()->thread_number()];
        for(int i=0; i < Config::instance()->thread_number(); i++){
            threads[i] = std::thread(inference, i, all_data + i * Config::instance()->thread_buffer_cnt(), this);
        }
        std::cout << "Threads started, wait for all threads end." << std::endl;
        for(int i=0; i < Config::instance()->thread_number(); i++){
            threads[i].join();
        }
    }
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    std::cout << "The total run time including compile is: " << elapsed_seconds.count() << "s" << std::endl;
    delete[] all_data;

    finish();
}