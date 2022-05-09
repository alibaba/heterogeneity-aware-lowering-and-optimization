<!-- markdown-link-check-disable -->
- [1.配置HALO、CUDA环境](#1配置halocuda环境)
  - [下载HALO](#下载halo)
  - [安装依赖](#安装依赖)
- [2.导出yolov5l模型](#2导出yolov5l模型)
  - [下载源码](#下载源码)
  - [导出yolov5l.onnx](#导出yolov5lonnx)
  - [导出旧版yolov5l.onnx](#导出旧版yolov5lonnx)
  - [注意事项](#注意事项)
- [3.编译模型](#3编译模型)
  - [编译yolov5l.onnx](#编译yolov5lonnx)
  - [运行模型](#运行模型)
  - [注意事项](#注意事项-1)

​		本文档主要说明使用HALO将yolov5l.onnx模型编译成.so文件，并基于该.so文件对图片、视频进行inference产出数据，然后对得出的数据进行成像处理。

## 1.配置HALO、CUDA环境
### 下载HALO
- 本例中使用基于release版本runtime的镜像, 并在该镜像的基础上安装依赖, 构建运行模型的镜像环境
```shell
docker pull reg.docker.alibaba-inc.com/aisml/sinianai_devl_env_cuda11.4_cudnn8_ubuntu18.04:latest-dev
```
- 另外也可以下载runtime包, 解压设置环境后即可使用
```shell
tar -xf HALO-{vesion}-Linux.tar.bz2 -C /opt/
cd /opt
mv HALO-{version}-Linux halo

echo "export PATH=$PATH:/opt/halo/bin" >> /etc/profile
echo "export LD_LIBRARAY_PATH=$LD_LIBRARAY_PATH:/opt/halo/lib" >> /etc/profile
source /etc/profile
```

### 安装依赖
- 需要满足的依赖
```shell
pip3 install --upgrade pip
pip3 install scipy==1.5.4 opencv-python==4.5.5.64 numpy  Pillow
apt-get update && apt-get install -y --no-install-recommends wget software-properties-common ffmpeg libsm6=2:1.2.2-1 libxext6=2:1.3.3-1  
```

- 安装cuda的runtime环境, 以下是通过网络的方式，先安装[cuda toolkit](https://developer.nvidia.com/cuda-downloads)
```shell
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /"
apt-get update
apt-get -y install cuda
```
  - 安装TensorRT C++应用

```shell
apt-get install libnvinfer8 libnvonnxparsers8 libnvparsers8 libnvinfer-plugin8
```
  - 验证安装

```shell
dpkg -l | grep TensorRT
```



## 2.导出yolov5l模型

### 下载源码
```shell
git clone https://github.com/ultralytics/yolov5
```

### 导出yolov5l.onnx
- 即使切换到这个版本，下载的也是最新版本的yolov5l模型，需要tag才可以下载指定版本。
- 该demo中使用的model为最新版本，即**v6.1**，如果下载的不是该版本，请参照下方的导出旧版本方式下载。
```shell
cd yolov5
git checkout -b tmp 7043872
PYTHONPATH=`pwd` python3 export.py --include onnx \
	--weights https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5l.pt \
	--img 640 --batch 1
```
- 以上执行完, 会在当前文件下保存两个文件yolov5l.pt和yolov5l.onnx。

### 导出旧版yolov5l.onnx

- 比如下面需要导出v5.0，可参照执行下面命令

```shell
cd yolov5
git checkout -b tmp 7043872
PYTHONPATH=`pwd` python3 export.py --include onnx \
	--weights https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5l.pt \
	--img 640 --batch 1
```

### 注意事项
- 在导出yolov5l.onnx之前需要配置相应的[依赖环境](https://github.com/ultralytics/yolov5/issues/251)。
- 版本run inference后结果数据的shape不同，以下脚本针对v6.1版本模型进行处理。
    - v6.1, (1, 25200, 85)
    - v5.0, (1, 3, 80, 80, 85)




## 3.编译模型
### 编译yolov5l.onnx
以下执行目录为/host/yolov5为参考，需要先将下载后的yolov5l.onnx模型拷贝到指定路径下。

- 编译

```shell
/opt/halo/bin/halo /host/yolov5/model/yolov5l.onnx -target cxx -entry-func-name=model -o /host/yolov5/out/yolov5l.cc

g++ /host/yolov5/out/yolov5l.cc -g -c -fPIC -I/opt/halo/include -o /host/yolov5/out/yolov5l.o
```
- 链接
```shell
g++ /host/yolov5/out/yolov5l.o /host/yolov5/out/yolov5l.bin -shared -lodla_tensorrt -g -Wl,-rpath=/opt/halo/lib -L /opt/halo/lib -o /host/yolov5/out/yolov5l.so
```

### 运行模型
- 在python脚本中加载yolov5l.so文件, 对输入的数据run inference。
- 配置环境依赖完之后, 运行脚本, 稍等些许时间, 即可在out文件中查看输出的图像或者视频文件。生成的图像可与使用onnx处理的比较。

```shell
bash run_yolov5.sh
```

### 注意事项

- 模型文件，脚本中引用$MODELS_ROOT/vision/detection/yolo/yolov5l.onnx。

- 输入文件，包含图片（$MODELS_ROOT/vision/detection/yolo/）、coco.names类名文件（../coco_classes.txt）。
- 输出文件，包含yolov5l.bin、yolov5l.cc、yolov5l.h、yolov5l.o、yolov5l.so、数据文件、效果图等。
