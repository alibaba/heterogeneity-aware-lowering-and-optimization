#include <ODLA/odla.h>
#include "model.h"
#include "cnpy.h"
#include "common.h"

using namespace std;

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

int main(int argc, char* argv[]){
  if(argc < 3)
    throw std::invalid_argument("Must have 2 parameters: --config <config.json>");
  std::string param(argv[1]);
  std::string config_file(argv[2]);
  if(param != "--config")
    throw std::invalid_argument("Usage --config <config.json>");
  MLPerfTest test;
  test.start(config_file);
  model_fini();
}
