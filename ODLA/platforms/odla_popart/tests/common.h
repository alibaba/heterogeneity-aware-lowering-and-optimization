#include <string>
#include <map>
#include <chrono>
#include <vector>
#include <ODLA/odla.h>
#include "json.hpp"
#include "cnpy.h"

#ifndef __COMMON__H_
#define __COMMON__H_

struct Float {
  // TODO(unknown): no infinity, underflow/overflow handling.
  Float() = delete;
  static constexpr int BitsPerByte = 8;
  template <typename T, int exp, int mantissa>
  static std::array<int, 3> Extract(T x) {
    static_assert(exp + mantissa + 1 == sizeof(T) * BitsPerByte);
    int sign = x >> (exp + mantissa);
    int m = x & ((1 << mantissa) - 1);
    int e = (x >> mantissa) & ((1 << exp) - 1);
    return {sign, e, m};
  }

  template <typename T, int exp, int mantissa>
  static T Combine(int sign, int e, int m) {
    static_assert(exp + mantissa + 1 == sizeof(T) * BitsPerByte);
    T x{0};
    x = sign ? 1U << (exp + mantissa) : 0;
    m >>= 32 - mantissa;
    x |= m & ((1U << mantissa) - 1);
    x |= (e & ((1U << exp) - 1)) << mantissa;
    return x;
  }
  static constexpr int FP32Exp = 8;
  static constexpr int FP32Mantissa = 23;
  static constexpr int FP32ExpBias = 127;
  static constexpr int FP16Exp = 5;
  static constexpr int FP16Mantissa = 10;
  static constexpr int FP16ExpBias = 15;

  static float GetFP32(uint16_t x) {
    auto components = Extract<uint16_t, FP16Exp, FP16Mantissa>(x);
    components[1] -= FP16ExpBias;
    components[2] <<= 32 - FP16Mantissa;
    // Underflow.
    if (components[1] == -FP16ExpBias) {
      while (components[2] > 0) {
        --components[1];
        components[2] <<= 1;
      }
      components[2] <<= 1;
    }
    return GetFP32(components[0], components[1], components[2]);
  }

  static float GetFP32(uint8_t sign, int32_t e, uint32_t m) {
    uint32_t x =
        Combine<uint32_t, FP32Exp, FP32Mantissa>(sign, e + FP32ExpBias, m);
    return *(reinterpret_cast<float*>(&x)); // NOLINT.
  }
};

/***************************************************************
 * The size of the parameters was defined by the npz file
 *   only read the npz file to get the data. So we don't need 
 *   the *batch_size* & *batches_per_step* information here.
 * 
 * Parameters needed for the run function:
 * {
 *    "model":"model_path_with_name",
 *    "thread_number":1,
 *    "duration":"40m",
 *    "npz":"npz_file_path_with_name",
 *    "inputs":
 *    {
 *      0:["indices", "UNIT32"], 
 *      1:["masks", "UNIT32"], 
 *      2:["positions", "UINT32"], 
 *      3:["segments", "UINT32"]
 *    },
 *    "outputs":
 *    {
 *        0:["squad_gemm", "FP16"]
 *        1:["someother", "FP16"]
 *    }
 * }
***************************************************************/
class Config{
private:
    std::string _call_method;
    std::uint32_t _dlopen_times;
    std::string _model_path;
    std::uint32_t _thread_number;
    std::uint32_t _duration;
    std::string _npz_file_path;
    std::map<std::string, std::vector<std::string>> _inputs;
    std::map<std::string, std::vector<std::string>> _outputs;
    std::string _result_file;
    std::uint32_t _thread_buffer_cnt;
    static Config* _instance;
    Config(){};
public:
    ~Config(){};
    static Config* instance(){return _instance;}
    const std::string& call_method(){return _call_method;}
    const std::uint32_t dlopen_times(){return _dlopen_times;}
    const std::string& model_path(){return _model_path;}
    const std::uint32_t thread_number(){return _thread_number;}
    const std::uint32_t thread_buffer_cnt(){return _thread_buffer_cnt;}
    const std::uint32_t duration(){return _duration;}
    const std::string& npz_file_path(){return _npz_file_path;}
    const std::string& result_file(){return _result_file;}
    const std::map<std::string, std::vector<std::string>>& inputs() {return _inputs;}
    const std::map<std::string, std::vector<std::string>>& outputs() {return _outputs;}
    void load(const std::string& file_path);
    void print();
};

class BaseTest{
public:
    BaseTest(){};
    ~BaseTest(){};
    void start(const std::string& config_file);
    virtual void do_inference(cnpy::npz_t& data) = 0;
    virtual void prerequisites() = 0;
    virtual void finish() = 0;
    void save_latency_results(const std::vector<float>& latencies, 
        const std::map<std::string, std::vector<cnpy::NpyArray>>& results); 
private:
    cnpy::npz_t* prepare_data();
};

extern void inference(int thread_id, cnpy::npz_t* all_data, BaseTest* test);

#endif//__COMMON__H_