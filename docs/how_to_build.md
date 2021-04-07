## System Requirements <a name="system-requirements"/>

HALO has been fully tested in the following development environment:

OS:
* Ubuntu 18.04

Tools and libraries:
* C++ compiler that supports C++17. (e.g. GCC >= 7.5.0)
* CMake (>= 3.14.5)
* Clang tools (>= 9.0)
* glog (>= 0.4)
* Protobuf 3.9.1

Software packages for some demos and examples:
* OpenCV 3.2.0
* Python3
* PyTorch and TensorFlow / Keras (to get pretrained model)
* ImageMagick (to preprocess test images)
* Device acceleration libraries:
  * [DNNL](https://github.com/oneapi-src/oneDNN)
  * [XNNPACK](https://github.com/google/XNNPACK)
  * [TensorRT](https://developer.nvidia.com/tensorrt)

NVIDIA® GPU environment:
* CUDA® (>= 10.0)
* CUDA® Deep Neural Network library™ (cuDNN) (>= 7.6.0)
* TensorRT™ (7.0.0)

## Docker Environment <a name="docker-environment"/>

For convenience, the above system requirements are also prepared and packed as a docker environment,
which is under [utils/docker](../utils/docker):

* [Dockerfile](../utils/docker/Dockerfile): contains all necessary software.
* [build_image.sh](../utils/docker/build_image.sh): it builds two docker images:
  * CPU-only: [ubuntu 18.04](https://hub.docker.com/_/ubuntu) based image;
  * CPU + GPU: [nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04](https://hub.docker.com/r/nvidia/cuda) based image.
* [start_docker_cpu.sh](../utils/docker/start_docker_cpu.sh): starts the CPU-only container.
* [start_docker_gpu.sh](../utils/docker/start_docker_gpu.sh): starts the container for CPU-only and CUDA® supported environments.

You can also pull the Docker images for development from aliyun:
```bash
docker pull registry-intl.us-west-1.aliyuncs.com/computation/halo:latest-devel-cuda10.0-cudnn7-ubuntu18.04 # Ubuntu 18.04 with CUDA 10.0

docker pull registry-intl.us-west-1.aliyuncs.com/computation/halo:latest-devel-x86_64-ubuntu18.04 # Ubuntu 18.04
```

## Build From Scratch <a name="build-from-scratch"/>

### Get HALO <a name="get-halo"/>
  ```bash
  git clone https://github.com/alibaba/heterogeneity-aware-lowering-and-optimization.git --recurse-submodules -j8
  ```
### Configure and Build <a name="configure-and-build">

```bash
mkdir halo/build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -G Ninja ..
ninja
```
Some CMAKE options:
* CMAKE_BUILD_TYPE=[Release|Debug]: select the build type.
* -DHALO_USE_GLOG=[ON]: use glob library for logging by default.
* -DHALO_CCACHE_BUILD=[ON] : enable or disable ccache for build.

### Unit Tests <a name="unit-tests"/>

HALO uses [llvm-lit](https://llvm.org/docs/CommandGuide/lit.html) test tools for unit testing. To run all unit tests, simply by

```bash
ninja check-halo
```

