#include <algorithm>
#include <array>
#include <fstream>
#include <iostream>
#include <vector>
#include <ctime>
#include <ODLA/odla.h>
#include <thread>

#include "model.h"
#include "cnpy.h"

using namespace std;

#define TOTAL_DATA_NUM 128//0

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

static cnpy::npz_t *all_data = nullptr;

void prepare_data()
{
  all_data = new cnpy::npz_t[TOTAL_DATA_NUM];
  cnpy::npz_t one_data = cnpy::npz_load("../data/bs10_bps128.npz"); 
  for(int i=0; i < TOTAL_DATA_NUM; i++)
    all_data[i] = one_data;
}

void inference(int start, int count, cnpy::npz_t* all_data){
  for(int i=start; i < start + count; i++){
    cnpy::npz_t* data = (all_data)+i;
    auto start_t = std::chrono::steady_clock::now();
	  model((*data)["indices"].data<std::uint32_t>(), 
            (*data)["masks"].data<std::uint32_t>(),
            (*data)["positions"].data<std::uint32_t>(),
            (*data)["segments"].data<std::uint32_t>(), 
            (*data)["squad_gemm"].data<std::uint16_t>()); 
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start_t;
    std::cout << "[" << i << "] @@@@@ request latency is: " << elapsed_seconds.count() << "s" << std::endl;
  }
}

#define THREAD_COUNT 32

int main(){

  prepare_data();
  bool single_thread = true;

  auto start = std::chrono::steady_clock::now();
  if(single_thread){
    for (int step=0; step<TOTAL_DATA_NUM; step++){
      cnpy::npz_t* data = all_data+step;
	    if(100 == step)
		    start = std::chrono::steady_clock::now();

      model((*data)["indices"].data<std::uint32_t>(), 
            (*data)["masks"].data<std::uint32_t>(),
            (*data)["positions"].data<std::uint32_t>(),
            (*data)["segments"].data<std::uint32_t>(), 
            (*data)["squad_gemm"].data<std::uint16_t>()); 
	  }
  }
  else{
    std::thread threads[THREAD_COUNT];
    int start_index = 0;
    int count = TOTAL_DATA_NUM/THREAD_COUNT;
    for(int i=0; i < THREAD_COUNT; i++){
      threads[i] = std::thread(inference, start_index, count, all_data);
      start_index += count;
    }
    cout << "Threads started, wait for all threads end." << std::endl;
    for(int i=0; i < THREAD_COUNT; i++){
      threads[i].join();
    }
  }
  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> elapsed_seconds = end-start;
  std::cout << "The total run time including compile is: " << elapsed_seconds.count() << "s" << std::endl;
  model_fini();
}
