# PyTorch Models
FROM pytorch/pytorch AS PT
RUN pip3 install scipy
RUN apt update && apt install -y wget imagemagick
COPY scripts/get_cls_model_from_pytorch.py /

WORKDIR /models/vision/classification/alexnet
RUN /get_cls_model_from_pytorch.py alexnet

WORKDIR /models/vision/classification/densenet
RUN /get_cls_model_from_pytorch.py densenet121

WORKDIR /models/vision/classification/googlenet
RUN /get_cls_model_from_pytorch.py googlenet

WORKDIR /models/vision/classification/inception
RUN /get_cls_model_from_pytorch.py inception_v3 inception_v3.onnx 299

WORKDIR /models/vision/classification/mobilenet
RUN /get_cls_model_from_pytorch.py mobilenet_v2

WORKDIR /models/vision/classification/squeezenet
RUN /get_cls_model_from_pytorch.py squeezenet1_0
RUN /get_cls_model_from_pytorch.py squeezenet1_1

WORKDIR /models/vision/classification/vgg
RUN /get_cls_model_from_pytorch.py vgg16
RUN /get_cls_model_from_pytorch.py vgg19

COPY scripts/get_unet.py /tmp
WORKDIR /models/vision/segmentation/unet
RUN python3 /tmp/get_unet.py
RUN wget https://github.com/zhixuhao/unet/raw/master/img/0test.png
RUN convert 0test.png -resize 256x256 test.jpg && rm 0test.png

# HRNet
FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-devel AS HRNET
SHELL [ "/bin/bash", "-c"]
RUN mv /etc/apt/sources.list.d/{cuda,nvidia-ml}.list /tmp/
RUN apt-get update && apt-get install -y curl && \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub | apt-key add - && \
    mv /tmp/{cuda.list,nvidia-ml.list} /etc/apt/sources.list.d/
RUN apt-get update && apt-get install -y git wget libgeos-dev gcc \
    libglib2.0-dev libsm6 libxext6 libxrender-dev
RUN pip3 install EasyDict==1.7 opencv-python==3.4.8.29 shapely==1.6.4 Cython \
    pandas pyyaml json_tricks scikit-image yacs>=0.1.5 \
    tensorboardX==1.6 pycocotools gdown
WORKDIR /tmp
RUN mkdir -p images annot /models/vision/pose_estimation
RUN git clone --depth=1 https://github.com/leoxiaobin/deep-high-resolution-net.pytorch
RUN python3 -c "import gdown; gdown.download('https://drive.google.com/uc?id=1_wn2ifmoQprBrFvUCDedjPON4Y6jsN-v', '/tmp/pose_hrnet_w32_256x256.pth')"
RUN wget 'https://upload-images.jianshu.io/upload_images/1877813-ff9b9c6b0e013006.jpg?imageMogr2/auto-orient/strip|imageView2/2/w/1200/format/webp' \
    -O images/005808361.jpg
RUN cd deep-high-resolution-net.pytorch/lib/nms && \
    python3 setup_linux.py build_ext --inplace
ENV PYTHONPATH=/tmp/deep-high-resolution-net.pytorch/lib
COPY scripts/hrnet/hrnet_cfg.yaml scripts/hrnet/get_hrnet.py ./
COPY scripts/hrnet/test.json annot
RUN python3 get_hrnet.py --cfg hrnet_cfg.yaml TEST.MODEL_FILE pose_hrnet_w32_256x256.pth
RUN mv pose_hrnet_w32_256x256.onnx input.in output.in /models/vision/pose_estimation

#YOLOv5
WORKDIR /tmp
ARG YOLOV5L_URI=https://github.com/ultralytics/yolov5
ARG YOLOv5_VERSION=v6.1
RUN apt install -y mesa-utils libgl1-mesa-glx
RUN git clone ${YOLOV5L_URI}
WORKDIR yolov5
RUN git checkout -b tmp 7043872
RUN pip3 install -r requirements.txt  # base requirements
RUN pip3 install coremltools>=4.1 onnx>=1.8.1 scikit-learn==0.19.2
RUN PYTHONPATH=/tmp/yolov5 python3 export.py --include onnx \
    --weights ${YOLOV5L_URI}/releases/download/${YOLOv5_VERSION}/yolov5l.pt --img 640 --batch 1
WORKDIR /models/vision/detection/yolo
RUN mv /tmp/yolov5/yolov5l.onnx .

# ONNX models
FROM alpine/git:v2.30.1 AS ONNX
RUN apk add git-lfs
WORKDIR /tmp
RUN git clone https://github.com/onnx/models
WORKDIR /tmp/models
RUN git checkout -f -b tmp 94abdef
RUN git-lfs pull -X="" -I="caffenet-3.onnx"
RUN git-lfs pull -X="" -I="yolov3-10.onnx"
RUN git-lfs pull -X="" -I="inception-v1-9.onnx"
RUN git-lfs pull -X="" -I="resnet18-v1-7.onnx"
RUN git-lfs pull -X="" -I="resnet50-v2-7.onnx"
RUN git-lfs pull -X="" -I="resnet101-v2-7.onnx"
RUN git-lfs pull -X="" -I="shufflenet-9.onnx"
RUN git-lfs pull -X="" -I="bertsqad-10.onnx"

WORKDIR /models/vision/classification/caffenet
RUN mv /tmp/models/vision/classification/caffenet/model/caffenet-3.onnx .

WORKDIR /models/vision/classification/inception
RUN mv /tmp/models/vision/classification/inception_and_googlenet/inception_v1/model/inception-v1-9.onnx .

WORKDIR /models/vision/classification/resnet
RUN mv /tmp/models/vision/classification/resnet/model/resnet50-v2-7.onnx .
RUN mv /tmp/models/vision/classification/resnet/model/resnet18-v1-7.onnx .
RUN mv /tmp/models/vision/classification/resnet/model/resnet101-v2-7.onnx .

WORKDIR /models/vision/classification/shufflenet
RUN mv /tmp/models/vision/classification/shufflenet/model/shufflenet-9.onnx .

WORKDIR /models/vision/detection/yolo
RUN mv /tmp/models/vision/object_detection_segmentation/yolov3/model/yolov3-10.onnx .

WORKDIR /models/text/comprehension/bert
RUN mv /tmp/models/text/machine_comprehension/bert-squad/model/bertsquad-10.onnx .

RUN cd /tmp/models && \
    git checkout -b yolov4 625eaec && \
    git-lfs pull -X="" -I="yolov4.onnx"

WORKDIR /models/vision/detection/yolo
RUN mv /tmp/models/vision/object_detection_segmentation/yolov4/model/yolov4.onnx .

WORKDIR /tmp
RUN git clone --depth=1 -b rel-1.8.0 https://github.com/onnx/onnx.git
RUN mv onnx/onnx/backend/test/data/node /unittests


# MNIST Simple TF Model
FROM tensorflow/tensorflow:1.14.0 AS TF
RUN apt-get install -y wget
WORKDIR mnist_simple
COPY scripts/mnist_simple_train.py .
RUN python mnist_simple_train.py
WORKDIR /models/vision/classification/mnist_simple
RUN mv /mnist_simple/mnist_simple.pb .
RUN wget -qO- https://web.archive.org/web/20160828233817/http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz  | gunzip -c > test_image
RUN wget -qO- https://web.archive.org/web/20160828233817/http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz | gunzip -c > test_label

# Data
FROM curlimages/curl:7.76.0 As Data
RUN mkdir -p /tmp/models/vision/test_images
WORKDIR /tmp/models/vision/test_images
RUN curl -o plane.jpg http://images.cocodataset.org/test2017/000000030207.jpg
RUN curl -o food.jpg http://images.cocodataset.org/test2017/000000228503.jpg
RUN curl -o sport.jpg http://images.cocodataset.org/test2017/000000133861.jpg
RUN curl -o dog.jpg https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg
RUN mkdir -p /tmp/models/detection
WORKDIR /tmp/models/detection
RUN curl -o test.jpg http://images.cocodataset.org/test2017/000000133861.jpg
RUN mkdir -p /tmp/models/vision/classification/caffenet
WORKDIR /tmp/models/vision/classification/caffenet
RUN curl -o caffenet.prototxt https://raw.githubusercontent.com/BVLC/caffe/master/models/bvlc_reference_caffenet/deploy.prototxt
RUN curl -o caffenet.caffemodel http://dl.caffe.berkeleyvision.org/bvlc_reference_caffenet.caffemodel
RUN mkdir -p /tmp/models/vision/detection/yolo
WORKDIR /tmp/models/vision/detection/yolo
RUN curl -o test.jpg http://images.cocodataset.org/test2017/000000133861.jpg
RUN curl -O https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/person.jpg
RUN curl -O https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/zidane.jpg


# Collect all Models & Data
FROM alpine
RUN apk add tree
COPY --from=PT /models /models
COPY --from=ONNX /models /models
COPY --from=HRNET /models /models
COPY --from=TF /models /models
COPY --from=DATA /tmp/models /models
COPY --from=ONNX /unittests /unittests
ENTRYPOINT ["tree", "/models"]