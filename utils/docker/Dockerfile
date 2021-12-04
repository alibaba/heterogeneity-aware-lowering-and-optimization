# syntax=docker/dockerfile:experimental
# Build this image:  docker build -t halo:[version] .

ARG BASE_IMAGE

# cmake
FROM ubuntu:18.04 AS cmake
ARG CMAKE_VERSION=3.14.5
RUN apt-get update && apt-get install -y curl gcc g++ make zlib1g zlib1g-dev libcurl4-openssl-dev git && \
    gcc --version && \
    mkdir /tmp/cmake && \
    cd /tmp/cmake && \
    curl -L https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}.tar.gz -o cmake.tar.gz && \
    tar zxf cmake.tar.gz && \
    cd cmake-${CMAKE_VERSION} && \
    ./bootstrap --system-curl --parallel=48 && \
    make -j all && \
    make install DESTDIR=/tmp/cmake/install && \
    make install && \
    tar -C /tmp/cmake/install -cf /tmp/cmake.tar usr && \
    rm -fr /tmp/cmake

# binutils
FROM cmake AS binutils
ARG BINUTILS_VERSION=2.35
RUN mkdir /tmp/binutils && \
    cd /tmp/binutils && \
    curl -s http://ftp.gnu.org/gnu/binutils/binutils-${BINUTILS_VERSION}.tar.gz | tar xvz && \
    cd binutils-${BINUTILS_VERSION} && \
    ./configure && \
    make -j all && \
    make install DESTDIR=/tmp/binutils/install && \
    tar -C /tmp/binutils/install -cf /tmp/binutils.tar usr && \
    rm -rf /tmp/binutils

# valgrind
FROM cmake AS valgrind
ARG VALGRIND_VERSION=3.18.1
RUN mkdir /tmp/valgrind && \
    cd /tmp/valgrind && \
    curl -o valgrind.tar.bz2 ftp://sourceware.org/pub/valgrind/valgrind-${VALGRIND_VERSION}.tar.bz2 && \
    tar jxf valgrind.tar.bz2 && \
    cd valgrind-${VALGRIND_VERSION} && \
    ./configure && \
    make -j all && \
    make install DESTDIR=/tmp/valgrind/install && \
    tar -C /tmp/valgrind/install -cf /tmp/valgrid.tar usr && \
    rm -rf /tmp/valgrind

# Build Protobuf (static)
FROM cmake AS pb
RUN git -C /tmp clone --depth=1 https://github.com/protocolbuffers/protobuf.git -b v3.9.1 && \
    cd /tmp/protobuf/cmake && \
    cmake -G "Unix Makefiles" -Dprotobuf_BUILD_TESTS=OFF \
      -Dprotobuf_BUILD_SHARED_LIBS=OFF \
      -DCMAKE_POSITION_INDEPENDENT_CODE=ON . && \
    make -j && make install DESTDIR=/tmp/protobuf/install && \
    tar -C /tmp/protobuf/install -cf /tmp/protobuf.tar usr && \
    rm -fr /tmp/protobuf

# Build Flatbuffer
FROM cmake as fb
RUN git -C /tmp clone --depth=1 https://github.com/google/flatbuffers.git -b v1.12.0 && \
    cd /tmp/flatbuffers && \
    cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release  -DFLATBUFFERS_BUILD_SHAREDLIB=OFF && \
    make -j && make install DESTDIR=/tmp/flatbuffers/install && \
    tar -C /tmp/flatbuffers/install -cf /tmp/flatbuffers.tar usr && \
    rm -fr /tmp/flatbuffers

# Build glog
FROM cmake AS glog
RUN git -C /tmp clone --depth=1 https://github.com/google/glog.git -b v0.4.0 && \
    cd /tmp/glog && \
    cmake -H. -Bbuild -G "Unix Makefiles" && \
    cd build && \
    make -j && make install DESTDIR=/tmp/glog/install && \
    tar -C /tmp/glog/install -cf /tmp/glog.tar usr && \
    rm -fr /tmp/glog

# Build DNNL
FROM cmake as dnnl
RUN git -C /tmp clone --depth=1 https://github.com/oneapi-src/oneDNN.git --branch v1.7 && \
    cd /tmp/oneDNN && \
    cmake -G "Unix Makefiles" -DDNNL_BUILD_EXAMPLES=OFF -DDNNL_BUILD_TESTS=OFF -DDNNL_ENABLE_PRIMITIVE_CACHE=ON -DCMAKE_INSTALL_PREFIX=/opt/dnnl && \
    make -j && make install DESTDIR=/tmp/oneDNN/install && \
    tar -C /tmp/oneDNN/install -cf /tmp/dnnl.tar opt && \
    rm -fr /tmp/oneDNN

# Build XNNPack
FROM cmake as xnnpack
RUN git -C /tmp clone https://github.com/google/XNNPACK.git && \
    cd /tmp/XNNPACK && git checkout -b tmp  90db69f681ea9abd1ced813c17c00007f14ce58b && \
    mkdir /tmp/xnn_build && cd /tmp/xnn_build && \
    cmake -G "Unix Makefiles" ../XNNPACK -DXNNPACK_LIBRARY_TYPE=static -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
      -DXNNPACK_BUILD_TESTS=OFF -DXNNPACK_BUILD_BENCHMARKS=OFF -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/opt/XNNPACK && \
    make -j && make install DESTDIR=/tmp/XNNPACK/install && \
    tar -C /tmp/XNNPACK/install -cf /tmp/xnnpack.tar opt && \
    rm -fr /tmp/XNNPACK /mp/xnn_build

FROM ${BASE_IMAGE}

# Redeclare the argument
ARG BASE_IMAGE

ARG python=3.7.0
ENV PYTHON_VERSION=${python}

# To access the host directory
RUN mkdir /host

RUN apt-get update && apt-get -y --no-install-recommends install software-properties-common apt-utils wget && rm -fr /var/lib/apt/lists/*

ARG GCC_VERSION=7
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

# Install TensorRT
ARG TENSORRT_VERSION=8.2.1-1+cuda11.4
RUN if [[ "${BASE_IMAGE}" =~ "nvidia" ]]; then apt-get update -y && \
    apt-get install -y --no-install-recommends --allow-change-held-packages \
        libnvinfer8=${TENSORRT_VERSION} \
        libnvinfer-dev=${TENSORRT_VERSION} \
        libnvinfer-plugin8=${TENSORRT_VERSION} \
        libnvonnxparsers8=${TENSORRT_VERSION} \
        python3-libnvinfer-dev=${TENSORRT_VERSION} \
        python3-libnvinfer=${TENSORRT_VERSION} \
        libnvinfer-plugin-dev=${TENSORRT_VERSION} \
        libnvparsers-dev=${TENSORRT_VERSION} \
        libnvonnxparsers-dev=${TENSORRT_VERSION} \
        libnvparsers8=${TENSORRT_VERSION} && \
    apt-get clean && apt-get purge && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* ; fi

# INSTALL LLVM
ARG LLVM_VERSION=9
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
RUN --mount=from=binutils,target=/pkg,source=/tmp tar -C / -xf /pkg/binutils.tar

# Install cmake
RUN --mount=from=cmake,target=/pkg,source=/tmp tar -C / -xf /pkg/cmake.tar

# Install valgrind
RUN --mount=from=valgrind,target=/pkg,source=/tmp tar -C / -xf /pkg/valgrid.tar

# INSTALL Protobuf (static)
RUN --mount=from=pb,target=/pkg,source=/tmp tar -C / -xf /pkg/protobuf.tar

# INSTALL glog
RUN --mount=from=glog,target=/pkg,source=/tmp tar -C / -xf /pkg/glog.tar

# Install GCC-10
RUN add-apt-repository ppa:ubuntu-toolchain-r/test && \
    apt-get update && \
    apt-get install -y gcc-10 g++-10 -y --no-install-recommends && \
    apt-get clean && apt-get purge && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Build & Install DNNL (MKLDNN)
RUN --mount=from=dnnl,target=/pkg,source=/tmp tar -C / -xf /pkg/dnnl.tar

# Install Parallel
RUN apt-get update && \
    apt-get install -y parallel --no-install-recommends && \
    apt-get clean && apt-get purge && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install Eigen
RUN curl -s https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.bz2 | tar -C /opt -xvj

# Install XNNPack
RUN --mount=from=xnnpack,target=/pkg,source=/tmp tar -C / -xf /pkg/xnnpack.tar

# Install Flatbuffer
RUN --mount=from=fb,target=/pkg,source=/tmp tar -C / -xf /pkg/flatbuffers.tar

# INSATLL ONEAPI
RUN if [[ ! "${BASE_IMAGE}" =~ "nvidia" ]]; then wget https://registrationcenter-download.intel.com/akdlm/irc_nas/17769/l_BaseKit_p_2021.2.0.2883_offline.sh && \
    chmod +x ./l_BaseKit_p_2021.2.0.2883_offline.sh && \
    ./l_BaseKit_p_2021.2.0.2883_offline.sh -s -a --silent --eula accept; fi

# Install models & test cases
COPY --from=registry-intl.us-west-1.aliyuncs.com/computation/halo:latest-model-zoo /models /models
COPY --from=registry-intl.us-west-1.aliyuncs.com/computation/halo:latest-model-zoo /unittests /unittests

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
