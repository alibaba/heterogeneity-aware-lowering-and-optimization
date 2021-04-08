# Build this image:  docker build -t halo:[version] .

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# Redeclare the argument
ARG BASE_IMAGE

ENV \
  CUDNN_VERSION=7.6.0.64-1+cuda10.0 \
  TENSORRT_VERSION=7.0.0-1+cuda10.0 \
  CMAKE_VERSION=3.14.5 \
  VALGRIND_VERSION=3.13.0 \
  GCC_VERSION=7 \
  LLVM_VERSION=9 \
  BINUTILS_VERSION=2.35

ARG python=3.7.0
ENV PYTHON_VERSION=${python}

# To access the host directory
RUN mkdir /host

RUN apt-get update && apt-get -y --no-install-recommends install software-properties-common apt-utils wget && rm -fr /var/lib/apt/lists/*

RUN apt-get update -y && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
      apt-transport-https \
      build-essential \
      autoconf \
      automake \
      libtool \
      cgdb \
      ccache \
      nasm \
      libc6-dbg \
      qemu-user \
      git-core \
      ca-certificates \
      gdb \
      vim \
      curl \
      libcurl4-openssl-dev \
      cpio \
      sudo \
      pkg-config \
      zip \
      zlib1g-dev \
      xterm \
      unzip \
      libpcre3 \
      libpcre3-dev \
      checkinstall \
      yasm \
      gfortran \
      libpng-dev \
      libjpeg8 \
      libjpeg8-dev \
      gpg-agent \
      graphviz \
      doxygen \
      python3-setuptools \
      python3-dev \
      python3-pip \
      openssh-client \
      openssh-server  \
      g++-${GCC_VERSION} gcc-${GCC_VERSION}  \
      g++-${GCC_VERSION}-aarch64-linux-gnu \
      less \
      scons \
      git-lfs \
      ninja-build \
      libopencv-core-dev \
      libopencv-highgui-dev \
      libopencv-videoio-dev \
      libmpc-dev \
      libmpfr-dev \
      libgmp-dev\
      gawk \
      imagemagick \
      bison \
      flex \
      texinfo \
      texlive \
      texlive-latex-extra \
      latex-cjk-all \
      libglib2.0-dev \
      libpixman-1-dev \
      bc && \
    apt-get clean && apt-get purge && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

#install the Git
RUN add-apt-repository ppa:git-core/ppa -y && apt update && apt install git -y

RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-${GCC_VERSION} 60 --slave /usr/bin/g++ g++ /usr/bin/g++-${GCC_VERSION} && \
    update-alternatives --install /usr/bin/aarch64-linux-gnu-g++ aarch64-linux-gnu-g++ /usr/bin/aarch64-linux-gnu-g++-${GCC_VERSION} 60 && \
    update-alternatives --install /usr/bin/aarch64-linux-gnu-gcc aarch64-linux-gnu-gcc /usr/bin/aarch64-linux-gnu-gcc-${GCC_VERSION} 60 && \
    update-alternatives --install "/usr/bin/ld" "ld" "/usr/bin/ld.gold" 20 && \
    update-alternatives --install "/usr/bin/ld" "ld" "/usr/bin/ld.bfd" 10

SHELL ["/bin/bash", "-c"]

#update/install the CUDNN
RUN if [[ "${BASE_IMAGE}" =~ "nvidia" ]]; then echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list ; fi
RUN if [[ "${BASE_IMAGE}" =~ "nvidia" ]]; then echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda.list ; fi
RUN if [[ "${BASE_IMAGE}" =~ "nvidia" ]]; then wget -qO - https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub | apt-key add - ;fi
RUN if [[ "${BASE_IMAGE}" =~ "nvidia" ]]; then wget -qO - https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub | apt-key add -; fi
RUN if [[ "${BASE_IMAGE}" =~ "nvidia" ]]; then add-apt-repository -y ppa:ubuntu-toolchain-r/test ; fi
RUN if [[ "${BASE_IMAGE}" =~ "nvidia" ]]; then apt-get update -y && apt-get install -y --no-install-recommends --allow-change-held-packages --allow-downgrades \
        libcudnn7=${CUDNN_VERSION} \
        libcudnn7-dev=${CUDNN_VERSION} ; fi

# Install TensorRT
RUN if [[ "${BASE_IMAGE}" =~ "nvidia" ]]; then apt-get update -y && \
    apt-get install -y --no-install-recommends --allow-change-held-packages \
        libnvinfer7=${TENSORRT_VERSION} \
        libnvinfer-dev=${TENSORRT_VERSION} \
        libnvinfer-plugin7=${TENSORRT_VERSION} \
        libnvonnxparsers7=${TENSORRT_VERSION} \
        python3-libnvinfer-dev=${TENSORRT_VERSION} \
        python3-libnvinfer=${TENSORRT_VERSION} \
        libnvinfer-plugin-dev=${TENSORRT_VERSION} \
        libnvparsers-dev=${TENSORRT_VERSION} \
        libnvonnxparsers-dev=${TENSORRT_VERSION} \
        libnvparsers7=${TENSORRT_VERSION} && \
    apt-get clean && apt-get purge && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* ; fi

# INSTALL LLVM
RUN wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add - && \
    add-apt-repository "deb http://apt.llvm.org/bionic/   llvm-toolchain-bionic-${LLVM_VERSION}  main"  && \
    apt-get update && apt-get install -y --no-install-recommends \
      clang-${LLVM_VERSION} \
      clangd-${LLVM_VERSION} \
      clang-tools-${LLVM_VERSION} \
      clang-tidy-${LLVM_VERSION} \
      clang-format-${LLVM_VERSION} && \
    apt-get clean && apt-get purge && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

#python
RUN pip3 install wheel numpy six jupyter enum34 mock h5py pillow

# Update binutils
RUN mkdir /tmp/binutils && \
    cd /tmp/binutils && \
    wget http://ftp.gnu.org/gnu/binutils/binutils-${BINUTILS_VERSION}.tar.gz && \
    tar zxf binutils-${BINUTILS_VERSION}.tar.gz && \
    cd binutils-${BINUTILS_VERSION} && \
    ./configure && \
    make -j all && \
    make install && \
    rm -rf /tmp/binutils

# Install cmake
RUN mkdir /tmp/cmake && \
    cd /tmp/cmake && \
    wget https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}.tar.gz && \
    tar zxf cmake-${CMAKE_VERSION}.tar.gz && \
    cd cmake-${CMAKE_VERSION} && \
    ./bootstrap --system-curl --parallel=48 && \
    make -j all && \
    make install && \
    rm -rf /tmp/cmake

# Install valgrind
RUN mkdir /tmp/valgrind && \
    cd /tmp/valgrind && \
    wget ftp://sourceware.org/pub/valgrind/valgrind-${VALGRIND_VERSION}.tar.bz2 && \
    tar jxf valgrind-${VALGRIND_VERSION}.tar.bz2 && \
    cd valgrind-${VALGRIND_VERSION} && \
    ./configure && \
    make -j all && \
    make install && \
    rm -rf /tmp/valgrind

# INSTALL Protobuf
RUN cd /tmp && \
    git clone --depth=1 https://github.com/protocolbuffers/protobuf.git -b v3.9.1 && \
    cd protobuf && \
    ./autogen.sh && ./configure && make -j && make install && \
    rm -fr /tmp/protobuf && \
    echo "/usr/local/lib" >> /etc/ld.so.conf

# INSTALL glog
RUN cd /tmp && \
    git clone --depth=1 https://github.com/google/glog.git -b v0.4.0 && \
    cd glog && \
    cmake -H. -Bbuild -G "Unix Makefiles" && cmake --build build && \
    cmake --build build --target install && ldconfig && \
    rm -fr /tmp/glog

# Install GCC-10
RUN add-apt-repository ppa:ubuntu-toolchain-r/test && \
    apt-get update && \
    apt install gcc-10 g++-10 -y --no-install-recommends && \
    apt-get clean && apt-get purge && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Build & Install DNNL (MKLDNN)
RUN cd /tmp && git clone --depth=1 https://github.com/oneapi-src/oneDNN.git --branch v1.7 && \
    cd /tmp/oneDNN && \
    cmake -G Ninja -DDNNL_BUILD_EXAMPLES=OFF -DDNNL_BUILD_TESTS=OFF -DDNNL_ENABLE_PRIMITIVE_CACHE=ON -DCMAKE_INSTALL_PREFIX=/opt/dnnl && \
    ninja install

# Install Eigen
RUN cd /tmp && wget https://gitlab.com/libeigen/eigen/-/archive/3.3.7/eigen-3.3.7.tar.bz2 && \
    tar jxvf eigen-3.3.7.tar.bz2 && mv eigen-3.3.7 /opt

# Install XNNPack
RUN cd /tmp && git clone https://github.com/google/XNNPACK.git && \
    cd /tmp/XNNPACK && git checkout -b tmp  90db69f681ea9abd1ced813c17c00007f14ce58b && \
    mkdir /tmp/xnn_build_static && cd /tmp/xnn_build_static && \
    cmake -G Ninja ../XNNPACK -DXNNPACK_LIBRARY_TYPE=static -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
      -DXNNPACK_BUILD_TESTS=OFF -DXNNPACK_BUILD_BENCHMARKS=OFF -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/opt/XNNPACK && \
    ninja install

# Install frameworks for model conversion
RUN pip3 install torch
RUN pip3 install tensorflow tf_slim scipy matplotlib
RUN pip3 install EasyDict==1.7 opencv-python==3.4.1.15 shapely==1.6.4 Cython \
    pandas pyyaml json_tricks scikit-image yacs>=0.1.5 \
    tensorboardX==1.6 torchvision pycocotools wget gdown GitPython

# Install Flatbuffer
RUN cd /tmp && \
    git clone --depth=1 https://github.com/google/flatbuffers.git -b v1.12.0 && \
    cd flatbuffers && \
    cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release  -DFLATBUFFERS_BUILD_SHAREDLIB=ON && make -j && make install && \
    rm -fr /tmp/flatbuffers

# INSATLL ONEAPI
RUN if [[ "${BASE_IMAGE}" != "nvidia" ]]; then wget https://registrationcenter-download.intel.com/akdlm/irc_nas/17769/l_BaseKit_p_2021.2.0.2883_offline.sh && \
    chmod +x ./l_BaseKit_p_2021.2.0.2883_offline.sh && \
    ./l_BaseKit_p_2021.2.0.2883_offline.sh -s -a --silent --eula accept; fi

RUN mkdir -p /var/run/sshd
RUN sed -i 's/prohibit-password/yes/' /etc/ssh/sshd_config

# Allow OpenSSH to talk to containers without asking for confirmation
RUN echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config

# Set SSH with the deploy key
ENV SSHDIR /root/.ssh
RUN mkdir -p ${SSHDIR}
RUN echo "StrictHostKeyChecking no" > ${SSHDIR}/config

# Add PATH
RUN echo "PATH=\".:/usr/local/cuda/bin:\$PATH\"" >> /root/.profile

#Clean up
RUN apt clean && apt purge && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Start the ssh
ENTRYPOINT service ssh restart && ldconfig && bash
