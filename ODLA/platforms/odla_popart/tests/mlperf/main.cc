#include <algorithm>
#include <array>
#include <fstream>
#include <iostream>
#include <vector>
#include <ctime>
#include <ODLA/odla.h>
#include <thread>
#include <cstring>
#include <iomanip>

#include "model.h"
#include "cnpy.h"
#include "common.h"

using namespace std;

//static cnpy::npz_t *all_data = nullptr;
class MLPerfTest : public BaseTest{
public:
  MLPerfTest(){};
  ~MLPerfTest(){};
  void do_inference(cnpy::npz_t& data) override
  {
    model(data["indices"].data<std::uint32_t>(), 
          data["masks"].data<std::uint32_t>(),
          data["positions"].data<std::uint32_t>(),
          data["segments"].data<std::uint32_t>(), 
          data["squad_gemm"].data<odla_float16>()); 
  }
};

int main(){
  MLPerfTest test;
  test.start();
  model_fini();
}
