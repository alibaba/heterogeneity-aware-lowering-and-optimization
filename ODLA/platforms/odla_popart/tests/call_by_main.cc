#include <ODLA/odla.h>
#include <dlfcn.h>
#include <iostream>
#include "model.h"
#include "cnpy.h"
#include "common.h"

using namespace std;

class CallByMainTest : public BaseTest{
public:
  CallByMainTest(){};
  ~CallByMainTest(){};
  void do_inference(cnpy::npz_t& data) final
  {
    int num_inputs = Config::instance()->inputs().size();
    int num_outputs = Config::instance()->outputs().size();
    const void** inputs = new const void*[num_inputs];
    void** outputs = new void*[num_outputs];
    for(auto &input : Config::instance()->inputs())
    {
      int idx = std::stoi(input.second[0]);
      inputs[idx] = data[input.first].data<unsigned char>();  //no matter the type of the data, just return a pointer to it
    }
    for(auto &output : Config::instance()->outputs())
    {
      int idx = std::stoi(output.second[0]);
      outputs[idx] = data[output.first].data<unsigned char>(); 
    }
    
    model_run(num_inputs, inputs, num_outputs, outputs, 0);  //The batch was not used

    delete[] inputs;
    delete[] outputs;
  }

  void prerequisites() final
  { 
  }

  void finish() final
  {
    model_fini();
  }
};

template<typename T>
static void check(T* h) {
    if(!h) {
        std::cerr << dlerror();
        exit(1);
    }
}

int main(int argc, char* argv[]){
  if(argc < 3)
    throw std::invalid_argument("Must have 2 parameters: --config <config.json>");
  std::string param(argv[1]);
  std::string config_file(argv[2]);
  if(param != "--config")
    throw std::invalid_argument("Usage --config <config.json>");
  
  CallByMainTest test;
  test.start(config_file);
}
