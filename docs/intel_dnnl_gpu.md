# Support Intel(R) GPU

## Hardware

### Intel GPU List
HALO supports Intel(R) GPU by Intel(R) [oneDNN](https://github.com/oneapi-src/oneDNN). Please check the supported GPU list in oneDNN website.

Here is the list copied from oneDNN:

- Intel Processor Graphics based on Gen9, Gen9.5 and Gen11, and Gen12 architectures
- Intel Iris(R) Xe graphics (formerly DG1)
- Future Intel Arc(TM) graphics (code name Alchemist and DG2)

### Driver
Please refer the [Intel® software for general purpose GPU capabilities](https://dgpu-docs.intel.com/index.html) to install the driver and softwares.

### Check
In Ubuntu, please install tool:
```
sudo apt update
sudo apt install clinfo
```

Run the cmd to check the Intel GPU is ready:
```
clinfo -l
or
sudo clinfo -l
```

For Gen9, the result is:
```
Platform #0: Intel(R) OpenCL HD Graphics
 `-- Device #0: Intel(R) Graphics [0x5927]

```

## Build HALO to support Intel(R) GPU

### Set oneAPI Packages

```
source /opt/intel/oneapi/compiler/latest/env/vars.sh
source /opt/intel/oneapi/dnnl/latest/env/vars.sh
source /opt/intel/oneapi/dpl/latest/env/vars.sh
source /opt/intel/oneapi/tbb/latest/env/vars.sh
```

### Build HALO with Intel GPU

Add `-DODLA_BUILD_DNNL_GPU=ON` in cmake cmd:

```
cd heterogeneity-aware-lowering-and-optimization
mkdir build
cd build

cmake -DCMAKE_BUILD_TYPE=Release -DODLA_BUILD_POPART=OFF -DODLA_BUILD_TRT=OFF -DODLA_BUILD_DNNL_GPU=ON -G Ninja ..
ninja
```

There will be file `./lib/libodla_dnnl_gpu.so` to be created.

## Build Binary to Support Intel(R) GPU

Set path for binary and library:
```
export HALO_ROOT=/xxx/halo
export PATH=$HALO_ROOT/heterogeneity-aware-lowering-and-optimization/build/bin/:$PATH
export LD_LIBRARY_PATH=$DNNLROOT/lib:$HALO_ROOT/heterogeneity-aware-lowering-and-optimization/build/lib:$LD_LIBRARY_PATH
```

Add `-lodla_dnnl_gpu` in link cmd.

## Execute the Binary

If the build & running time environment are different, you would meet error to miss: libprotobuf.so.3.9.1.0
Please search and copy it from build environment to running time.


