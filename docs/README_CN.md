<!-- markdown-link-check-disable -->
- [1. Docker镜像](#1-docker镜像)
  - [1.1 runtime环境镜像](#11-runtime环境镜像)
  - [1.2 容器环境编译](#12-容器环境编译)
- [2. 源码编译方式安装](#2-源码编译方式安装)
  - [2.1 环境准备](#21-环境准备)
    - [安装GCC、Python、pip、Ninja等组件](#安装gccpythonpipninja等组件)
    - [安装CMake](#安装cmake)
    - [安装protobuf](#安装protobuf)
    - [安装flatbuffers](#安装flatbuffers)
    - [安装glog](#安装glog)
    - [安装Clang](#安装clang)
    - [安装TensorRT](#安装tensorrt)
    - [安装mkl-dnn](#安装mkl-dnn)
    - [安装eigen](#安装eigen)
    - [安装XNNPACK](#安装xnnpack)
    - [安装opencv4](#安装opencv4)
  - [2.2 <span id="download_resource">下载源码</span>](#22-下载源码)
  - [2.3 编译及测试](#23-编译及测试)
  - [2.4 解压安装](#24-解压安装)
  - [2.5 验证版本](#25-验证版本)


本文档介绍[HALO](https://github.com/alibaba/heterogeneity-aware-lowering-and-optimization)在Linux环境下的安装，可选**容器、源码**两种方式。基于源码编译安装需要的依赖较多，**推荐在容器环境下编译**，编译完成后将二进制文拷贝到您的执行环境中，当然我们也提供已构建完成的runtime镜像。

### 1. Docker镜像

| 硬件平台 |                          Docker仓库                          |                   标签                    |                  说明                   |
| :------: | :----------------------------------------------------------: | :---------------------------------------: | :-------------------------------------: |
|   GPU    |    registry-intl.us-west-1.aliyuncs.com/computation/halo     | 0.7.6-devel-cuda11.4.2-cudnn8-ubuntu18.04 |    提供基于ubuntu18.04构建halo的环境    |
|          |    registry-intl.us-west-1.aliyuncs.com/computation/halo     |        0.7.2-devel-x86_64-centos7         |      提供基于centos7构建halo的环境      |
|          | reg.docker.alibaba-inc.com/aisml/sinianai_devl_env_cuda11.4_cudnn8_ubuntu18.04 |                latest-dev                 | 提供基于ubuntu18.04环境运行时的二进制包 |
|          | reg.docker.alibaba-inc.com/aisml/sinianai_devl_env_cuda11.3_cudnn8_ubuntu18.04 |               latest-centos               |  提供基于centos7.6环境运行时的二进制包  |

#### 1.1 runtime环境镜像

- 根据所需环境拉取runtime镜像，开箱即用。

```shell
$ docker pull reg.docker.alibaba-inc.com/aisml/sinianai_devl_env_cuda11.3_cudnn8_ubuntu18.04:latest-dev
```

- 验证，可查看使用及相应版本。

```shell
$ docker run reg.docker.alibaba-inc.com/aisml/sinianai_devl_env_cuda11.3_cudnn8_ubuntu18.04:latest-dev halo -h
```

#### 1.2 容器环境编译

首先需要拉取最新的源码到有docker服务的主机上，拉取编译环境的镜像，将该源码的目录挂载到容器内，执行编译。

```shell
$ docker pull registry-intl.us-west-1.aliyuncs.com/computation/halo:0.7.6-devel-cuda11.4.2-cudnn8-ubuntu18.04
```

- 源码下载请参考源码编译方式安装中[下载源码](#download_resource)部分，进入到源码路径下，启动容器。

```shell
$ cd heterogeneity-aware-lowering-and-optimization
$ docker run --runtime=nvidia -e NVIDIA_DRIVER_CAPABILITIES=video,compute,utility \
	--privileged --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
	-it --name halo_build_env \
	-v `pwd`:/host --tmpfs /tmp:exec \
	--rm registry-intl.us-west-1.aliyuncs.com/computation/halo:0.8.0-devel-cuda11.4.2-cudnn8-ubuntu18.04
```

- 执行编译操作，每个步骤执行顺利到最后会生成`HALO-{version}-Linux.tar.bz2`的包，在容器环境中*验证与源码环境编译*类似。
- 如果您想定制自己的镜像环境，可参考[Dockerfile](https://github.com/alibaba/heterogeneity-aware-lowering-and-optimization/blob/master/utils/docker/Dockerfile)。

### 2. 源码编译方式安装

本部分基于GPU环境使用源码编译方式安装HALO，下面以Ubuntu18.04为例做编译安装说明。

- 如果您的系统已经安装了部分依赖，如CUDA，Python，GCC等，可参照下面的安装步骤手动安装。

#### 2.1 环境准备

下表列出了编译安装HALO所需的系统环境和第三方依赖。

|  软件名称   |     版本     |          作用          |
| :---------: | :----------: | :--------------------: |
|   Ubuntu    |    18.04     |  编译、运行的操作系统  |
|     git     |  2.0及以上   |     源代码管理工具     |
|   git-lfs   |      -       |  源代码大文件管理工具  |
|    gcc-7    |  7.0及以上   |       C++编译器        |
|   gcc-10    |     10.0     |       C++编译器        |
|    cmake    | 3.14.5及以上 |        编译工具        |
|    Ninja    | 1.8.2及以上  |      并行构建工具      |
|   ccache    |      -       |      编译缓存工具      |
|    Clang    |    9.0.0     |        编译工具        |
|   Python    |    3.6.0     |    依赖的python环境    |
|  protobuf   |    3.9.1     |    依赖的序列化工具    |
| flatbuffers |    1.12.0    | 依赖的跨平台序列化工具 |
|    glog     |    0.4.0     |      依赖的C++库       |
|   doxygen   |      -       |        文档工具        |

可选的AI加速库，用于构建相应的ODLA Runtime库, 并运行相应的测试和Demo.

|    软件名称     |         版本         |                             作用                             |
| :-------------: | :------------------: | :----------------------------------------------------------: |
|    TensorRT     |        8.2.1         |      构建 odla_tensorrt库, 从而在Nvidia GPU上运行模型。      |
|     mkl-dnn     |        1.7-rc        |        构建 odla_dnnl运行库，从而在x86 CPU 上运行模型        |
|      eigen      |        3.4.0         |      构建 odla_eigen运行库，从而利用Eign加速库运行模型       |
|     XNNPACK     | (git SHA: 90db69f68) | 构建 odla_xnnpack运行库，从而通过XNNPACK加速库在X86和ARM上运行模型 |
| 其它芯片厂商SDK |          -           | 用于构建相应的ODLA Runtime 库, 如可在寒武纪平台运行的 odla_magicmind. |

其它运行Demo所需的工具：

| 软件名称 | 版本  |          作用          |
| :------: | :---: | :--------------------: |
|  OpenCV  | 4.4.0 | 图像的预处理，后处理等 |
|   pip    |   -   | 使用的python包管理工具 |

##### 安装GCC、Python、pip、Ninja等组件

```shell
$ sudo apt-get install -y --no-install-recommends software-properties-common
$ sudo add-apt-repository ppa:ubuntu-toolchain-r/test
$ sudo apt-get update
$ sudo apt-get install -y --no-install-recommends \
    g++-7 gcc-7 \
    g++-7-aarch64-linux-gnu \
    gcc-10 g++-10  \
    ninja-build \
    python3-pip python3-dev \
    doxygen nasm \
    libpng-dev libjpeg8 libjpeg8-dev
```

##### 安装[CMake](https://cmake.org/download/)

```shell
$ wget -q https://github.com/Kitware/CMake/releases/download/v3.16.7/cmake-3.16.7.tar.gz
$ tar zxf cmake-3.16.7.tar.gz
$ cd cmake-3.16.7
$ ./bootstrap --system-curl --parallel=48
$ make -j all
$ sudo make install
```

使用`cmake --version`查看CMake版本，若出现版本号，则表示安装成功。

##### 安装[protobuf](https://developers.google.com/protocol-buffers)

```shell
$ git clone --depth=1 https://github.com/protocolbuffers/protobuf.git -b v3.9.1
$ cd protobuf/cmake
$ cmake -G Ninja . -Dprotobuf_BUILD_TESTS=OFF \
	-Dprotobuf_BUILD_SHARED_LIBS=OFF \
	-DCMAKE_POSITION_INDEPENDENT_CODE=ON
$ sudo ninja install
```

使用`protoc --version`查看版本，若出现版本号，则表示安装成功。在Release模式下编译，推荐使用静态库，cmake时会使用`/usr/local/lib64/libprotobuf.a`，需要注意的是如果您的环境中存在`/usr/local/lib/libprotobuf.so`，该库会被优先使用。

##### 安装flatbuffers

```shell
$ git clone --depth=1 https://github.com/google/flatbuffers.git -b v1.12.0
$ cd flatbuffers
$ cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release -DFLATBUFFERS_BUILD_SHAREDLIB=ON
$ make -j8
$ sudo make install
```

##### 安装glog

```shell
$ git clone --depth=1 https://github.com/google/glog.git -b v0.4.0
$ cd glog
$ cmake -H. -Bbuild -G "Unix Makefiles"
$ cmake --build build
$ sudo cmake --build build --target install
```

##### 安装Clang

```shell
$ llvm_version=9
$ wget https://apt.llvm.org/llvm.sh
$ sudo bash ./llvm.sh ${llvm_version}
$ sudo apt-get install -y clang-tidy-${llvm_version}  clang-tools-${llvm_version} clang-format-${llvm_version}
```

##### 安装TensorRT

这里选择network的方式安装，更多方式请参照其[官网](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html)根据环境安装。

- 首先安装 [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)

```shell
  $ wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
  $ sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
  $ sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
  $ sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /"
  $ sudo apt-get update
  $ sudo apt-get -y install cuda
```

  - 再安装其他dev组件

```shell
$ TensorRT_version="8.2.1-1+cuda11.4"
$ sudo apt-get install -y --no-install-recommends --allow-change-held-packages \
	libnvinfer8=${TensorRT_version}  libnvinfer-dev=${TensorRT_version} \
	libnvinfer-plugin8=${TensorRT_version} libnvonnxparsers8=${TensorRT_version} \
	python3-libnvinfer-dev=${TensorRT_version} python3-libnvinfer=${TensorRT_version} \
	libnvinfer-plugin-dev=${TensorRT_version} libnvparsers-dev=${TensorRT_version} \
	libnvonnxparsers-dev=${TensorRT_version}  libnvparsers8=${TensorRT_version}
```

##### 安装mkl-dnn

```shell
$ git clone --depth=1 https://github.com/intel/mkl-dnn.git --branch v1.7-rc
$ cd /tmp/mkl-dnn
$ cmake -G Ninja -DDNNL_BUILD_EXAMPLES=OFF \
	-DDNNL_BUILD_TESTS=OFF \
	-DDNNL_ENABLE_PRIMITIVE_CACHE=ON \
	-DCMAKE_INSTALL_PREFIX=/opt/dnnl
$ sudo ninja install
```

##### 安装[eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page)

注意eigen这里的安装路径目前是固定在`/opt`下面。

```shell
$ wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.bz2
$ tar jxvf eigen-3.4.0.tar.bz2 -C /opt/
```

##### 安装[XNNPACK](https://github.com/google/XNNPACK.git)

```shell
$ git clone https://github.com/google/XNNPACK.git
$ cd XNNPACK
$ git checkout -b tmp 90db69f681ea9abd1ced813c17c00007f14ce58b
$ mkdir build && cd build
$ cmake -G Ninja .. -DXNNPACK_LIBRARY_TYPE=static -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
	-DXNNPACK_BUILD_TESTS=OFF -DXNNPACK_BUILD_BENCHMARKS=OFF \
	-DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/XNNPACK
$ sudo ninja install
```

##### 安装[opencv4](https://docs.opencv.org/4.x/d7/d9f/tutorial_linux_install.html)

```shell
$ OPENCV_VERSION=4.4.0
$ wget -O opencv.zip https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip
$ wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/${OPENCV_VERSION}.zip
$ unzip opencv.zip && unzip opencv_contrib.zip
$ cd opencv-${OPENCV_VERSION}
$ mkdir build && cd build
$ cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D INSTALL_PYTHON_EXAMPLES=ON \
      -D INSTALL_C_EXAMPLES=OFF -D WITH_TBB=ON -D WITH_CUDA=ON \
      -D BUILD_opencv_cudacodec=ON -D ENABLE_FAST_MATH=1 \
      -D CUDA_FAST_MATH=1 -D WITH_CUBLAS=1 -D WITH_V4L=ON -D WITH_QT=OFF -D WITH_OPENGL=ON \
      -D WITH_GSTREAMER=ON -D OPENCV_GENERATE_PKGCONFIG=ON -D OPENCV_PC_FILE_NAME=opencv.pc \
      -D OPENCV_ENABLE_NONFREE=ON -D OPENCV_PYTHON3_INSTALL_PATH=/usr/lib/python3.6/dist-packages \
      -D OPENCV_EXTRA_MODULES_PATH=../opencv_contrib-${OPENCV_VERSION}/modules \
      -D PYTHON_EXECUTABLE=/usr/bin/python -D BUILD_EXAMPLES=OFF \
      -D CUDA_ARCH_BIN=7.0 -DOPENCV_DNN_CUDA=OFF -DWITH_CUDNN=OFF ..
$ make -j
$ make install -j
```

- 安装其他芯片厂商的sdk（待更新...）

#### 2.2 <span id="download_resource">下载源码</span>

```shell
$ git clone --depth=1 https://github.com/alibaba/heterogeneity-aware-lowering-and-optimization.git
$ cd heterogeneity-aware-lowering-and-optimization
$ git submodule sync --recursive
$ git submodule update --init --recursive
$ git lfs install
$ git lfs pull
```

- 源码内容较大，拉取之前请保证本地磁盘有足够的剩余空间。
- 耗时较长，请耐心等待，若拉取失败，根据中断原因多尝试几次。

#### 2.3 编译及测试

进入HALO的源码根目录，创建相应的构建文件夹，执行一系列编译测试操作。

默认会编译下列**odla_runtime**库，可以选择使用`-DODLA_BUILD_XXX=OFF`关闭相应的库：

| 参数                  | 选项                 | 说明                      |
| :-------------------- | -------------------- | ------------------------- |
| ODLA_BUILD_EIGEN      | 可选ON或OFF，默认ON  | 构建 odla_eigen运行库     |
| ODLA_BUILD_DNNL       | 可选ON或OFF，默认ON  | 构建 odla_dnnl运行库      |
| ODLA_BUILD_POPART     | 可选ON或OFF，默认ON  | 构建 odla_popart运行库    |
| ODLA_BUILD_TRT        | 可选ON或OFF，默认ON  | 构建 odla_tensorrt库      |
| ODLA_BUILD_OPENVINO   | 可选ON或OFF，默认ON  | 构建 odla_openvino运行库  |
| ODLA_BUILD_HGAI       | 可选ON或OFF，默认ON  | 构建 odla_hgai运行库      |
| ODLA_BUILD_MAGICMIND  | 可选ON或OFF，默认OFF | 构建 odla_magicmind运行库 |
| ODLA_BUILD_XNNPACK    | 可选ON或OFF，默认ON  | 构建 odla_xnnpack运行库   |
| ODLA_BUILD_QAIC       | 可选ON或OFF，默认ON  | 构建 odla_qaic运行库      |
| ODLA_BUILD_Profiler   | 可选ON或OFF，默认ON  | 构建 odla_profiler运行库  |
| ODLA_BUILD_TF_Wrapper | 可选ON或OFF，默认OFF | --                        |

- 通常使用的构建方式：

```shell
$ cd heterogeneity-aware-lowering-and-optimization
$ mkdir build
$ cd build
$ cmake -G Ninja -DDEST_DIR=/tmp/halo_inst .. -L -DDNNL_COMPILER=gcc-10
$ ninja
$ ninja check-halo
$ ninja DOCS
$ ninja package
```

- 比如我们在centos环境中编译，不使用某些模块，可配置下面参数：

```shell
$ cmake -G Ninja -DDEST_DIR=/tmp/halo_inst .. -L -DDNNL_COMPILER=gcc-10 \
	-DODLA_BUILD_QAIC=OFF \
	-DODLA_BUILD_HGAI=OFF \
	-DODLA_BUILD_OPENVINO=OFF \
	-DODLA_BUILD_MAGICMIND=OFF
```

- `ninja check-halo`测试时间大约需要30分钟，请耐心等待。
- `ninja package`执行成功后会生成`HALO-{version}-Linux.tar.bz2`的包，里面包含halo运行的二进制文件、lib库等。

#### 2.4 解压安装

将编译完成后的tar包解压到指定路径，并添加环境变量后方可验证使用，比如下面以`/opt`为安装路径。

```shell
$ tar -xf HALO-{version}-Linux.tar.bz2 -C /opt
$ cd /opt
$ mv HALO-{version}-Linux halo
```

- 临时使用halo，在当前shell中执行。

```shell
$ export PATH="/opt/halo/bin:${PATH}"
$ export LD_LIBRARY_PATH="/opt/halo/lib:${LD_LIBRARY_PATH}"
```

- 若想到在其他终端也使用halo，需要加入系统中，可将该环境变量添加到/etc/profile文件内。

```shell
$ echo PATH="/opt/halo/bin:${PATH}" >> /etc/profile
$ echo LD_LIBRARY_PATH="/opt/halo/lib:${LD_LIBRARY_PATH}" >> /etc/profile
$ source /etc/profile
```

- 检查halo依赖的库是否加载或者存在，正常情况下会显示出来依赖库的路径`/opt/halo/lib`，假若执行报错需检查库是否完整。

```shell
$ ldconfig && ldconfig -v

$ ldd halo	# 检查当前环境下halo的依赖库是否存在缺失
```

#### 2.5 验证版本

添加完环境之后，在当前shell中执行，若出现下面内容即表示可用。

```shell
$ halo -h

USAGE: halo [options] model file name.

OPTIONS:

Generic Options:

  --help                              - Display available options (--help-hidden for more)
  --help-list                         - Display list of available options (--help-list-hidden for more)
  --version                           - Display the version of this program

Halo options:
	...
```
