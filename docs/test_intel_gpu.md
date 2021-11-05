# Test HALO on Intel® GPU

For **Hardware**, **building HALO and application** to support Intel® GPU, please refer to [Support Intel® GPU](intel_gpu.md)

## OS

**Ubuntu 20** is recommended. 

Ubuntu 18 could be used in building environment, but can't be used to run App, because the Intel GPU driver doesn't support Ubuntu 18.

## Prepare

### Check Hardware

Run the cmd to check the Intel GPU is ready:
```
clinfo -l
or
sudo clinfo -l
```

### Set oneAPI Packages

**Note**, please make sure the **release of oneAPI are same in running and build environments**, to avoid the possible compatibility issue.

```
source /opt/intel/oneapi/compiler/latest/env/vars.sh
source /opt/intel/oneapi/dnnl/latest/env/vars.sh
source /opt/intel/oneapi/dpl/latest/env/vars.sh
source /opt/intel/oneapi/tbb/latest/env/vars.sh
```

### Set path for HALO binary and library:
```
export HALO_ROOT=/xxx/halo
export PATH=$HALO_ROOT/heterogeneity-aware-lowering-and-optimization/build/bin/:$PATH
export LD_LIBRARY_PATH=$DNNLROOT/lib:$HALO_ROOT/heterogeneity-aware-lowering-and-optimization/build/lib:$LD_LIBRARY_PATH
```

### Run App

Please run the application built with ODLA.

### Tips
1. Error to miss so files when run App.

If the build & running environment are different, you would meet error to miss so files. Like: **libprotobuf.so.3.9.1.0**.

Please search and copy it from build environment to running time.


