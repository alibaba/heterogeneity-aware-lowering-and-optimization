#include <iostream>
#include <ODLA/odla.h>
#include <dlfcn.h>

/*****************************************************************
 * What parameters needed:
 * 1. model so path
 * 2. model init method name
 * 3. model run method name
 * 4. model fini method name
 * 5. destroyContext method name? Needed for multiple threads
 * 6. input number
 * 7. input tensorids?
 *****************************************************************/


extern "C" {
    odla_computation model_helper();
    void model_run(int num_inputs, const void* inputs[], void* outputs, int batch_size);
    void model_init();
    void model_fini();
}

template<typename T>
static void check(T* h) {
    if(!h) {
        std::cerr << dlerror();
        exit(1);
    }
}

int main() {
    float data32[] = {0};
    float buf [128];
    const void* inputs[1] = {data};
    void* outputs[1] = {buf};

    int cnt = 2;
    for(i=0; i < cnt; i++){
        auto h = dlopen("./tusou_model.so", RTLD_NOW | RTLD_GLOBAL);
        check(h);
        auto f = dlsym(h, "model_helper");
        check(f);

        decltype(&model_init) f_init = (decltype(f_init))(dlsym(h, "model_init"));
        check(f_init);
        decltype(&model_run) f_run = (decltype(f_run))(dlsym(h, "model_run"));
        check(f_run);
        decltype(&model_fini) f_fini = (decltype(f_fini))(dlsym(h, "model_fini"));
        check(f_fini);

        f_run(1, inputs, 1, outputs, 1);
        f_fini();
        dlclose(h);

        std::cout << "=========================== " << i << " finished" << std::endl;
    }
    std::cout << "===== Test Completed =====";
    return 0;
}