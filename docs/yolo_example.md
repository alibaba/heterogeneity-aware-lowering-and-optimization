### A Complete Example on Object Detection <a name="a-complete-example-on-object-detection"/>

This example demonstrates how to deploy a pretrained
[YOLO v3](https://pjreddie.com/darknet/yolo/) model with pre-processing and post-processing on host.

First, download the model from [ONNX Model Repository](https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/yolov3).
For demonstration purpose, we use [XNNPACK](https://github.com/google/XNNPACK)-based ODLA runtime, which
supports the ODLA interpret programming mode.

Next, compile the model into the C code:
```bash
halo -target cc -exec-mode=interpret -emit-value-id-as-int -reorder-data-layout=channel-last -remove-input-transpose -remove-output-transpose  -o out/yolo.c yolov3-10.onnx  --disable-broadcasting -outputs conv2d_59 -outputs conv2d_67 -outputs conv2d_75 -input-shape=input_1:1x3x416x416 -entry-func-name=yolo_v3
```
Options explained:
* `-target cc`: to generate the C99 code.
* `-exec-mode=interpret`: to generate the ODLA interpret mode code.
* `-emit-value-as-int`: to generate the ODLA value ids as integers.
* `-reorder-data-layout=channel-last`: to enable the data layout conversion since ONNX uses NCHW while the XNNPACK runtime prefers NHWC.
* `-remove-input-transpose`: to optimize away the input transpose.
* `-remove-output-transpose`: to optimize away the output transpose.
* `-disable-broadcasting`: to disable the offline weights broadcasting since the ODLA runtime supports element-wise ops with broadcasting.
* `-outputs`: to specify the output nodes by their names.
* `-input-shape`: to explicitly specify the input shape.
* `-entry-func-name=yolo_v3`: to specify the generate function names as yolo_v3(), yolo_v3_init() and yolo_v3_fini().
  
A complete Yolo application, including the input preprocessing, inferencing,
and the result rendering, can be found [here](models/vision/detection/yolo).

