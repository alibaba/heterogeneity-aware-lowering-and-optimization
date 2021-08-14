#include <ODLA/odla.h>
#include <dlfcn.h>
#include <iostream>
#include "model.h"
#include "cnpy.h"
#include "common.h"

using namespace std;

template<typename T>
static void check(T* h) {
    if(!h) {
        std::cerr << dlerror();
        exit(1);
    }
}

class CallByDlopenTest : public BaseTest{
public:
  CallByDlopenTest(){};
  ~CallByDlopenTest(){};
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
    
    _f_run(num_inputs, inputs, num_outputs, outputs, 0);  //The batch was not used
    delete[] inputs;
    delete[] outputs;
  }

  void prerequisites() final
  {
    _so_handle = dlopen(Config::instance()->model_path().c_str(), RTLD_NOW | RTLD_GLOBAL);
    check(_so_handle);
    auto f = dlsym(_so_handle, "model_helper");
    check(f);

    _f_init = (decltype(_f_init))(dlsym(_so_handle, "model_init"));
    check(_f_init);
    _f_run = (decltype(_f_run))(dlsym(_so_handle, "model_run"));
    check(_f_run);
    _f_fini = (decltype(_f_fini))(dlsym(_so_handle, "model_fini"));
    check(_f_fini);
    std::cout << "prerequisites done" << std::endl;
  }

  void finish() final
  {
    static int cnt = 0;
    std::cout << "finish() called " << ++cnt << " times." << std::endl;
    _f_fini();
    std::cout << "The computation deleted" << std::endl;
    dlclose(_so_handle);
    std::cout << "The " << Config::instance()->model_path() << " has been closed." << std::endl;
  }
private:
  void* _so_handle;
  decltype(&model_init) _f_init;
  decltype(&model_run) _f_run;
  decltype(&model_fini) _f_fini;
};

int main(int argc, char* argv[]){
  if(argc < 3)
    throw std::invalid_argument("Must have 2 parameters: --config <config.json>");
  std::string param(argv[1]);
  std::string config_file(argv[2]);
  if(param != "--config")
    throw std::invalid_argument("Usage --config <config.json>");
  
  CallByDlopenTest test;
  test.start(config_file);
  
}
