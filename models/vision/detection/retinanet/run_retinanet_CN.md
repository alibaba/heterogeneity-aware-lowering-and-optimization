<!-- markdown-link-check-disable -->
- [RetinaNet](#retinanet)
  - [数据准备](#数据准备)
  - [编译模型文件](#编译模型文件)
  - [运行inference](#运行inference)
  - [其他](#其他)

# RetinaNet

## 数据准备
- 使用模型中图片进行前置处理
```bash
input_image = Image.open(image_path)
prepro = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = prepro(input_image)
_preprocess_data = input_tensor.unsqueeze(0).numpy()
```

## 编译模型文件
- 根据模型文件input属性选择参数，编译生成retinanet-9.so库文件
```bash
halo retinanet-9.onnx -target cxx -disable-broadcasting -entry-func-name=model -o out/retinanet-9.cc --inputs=input
g++ ./out/retinanet-9.cc -g -c -fPIC -I/host/ODLA/include -o ./out/retinanet-9.o

g++ ./out/retinanet-9.o ./out/retinanet-9.bin -shared -lodla_tensorrt -g -Wl,-rpath=/host/build/lib -L /host/build/lib -o ./out/retinanet-9.so
```

## 运行inference
- 选择input数据的方式，默认从数据文件获取
- 根据shape构建输出数据
- 在一定误差范围内对比使用onnx的输出与使用halo处理后的输出
```bash
def struct_out(out):
    return (ctypes.c_float * reduce(lambda x, y: x * y, out))()

outputs_shape = [(1, 720, 60, 80), (1, 720, 30, 40), (1, 720, 15, 20), (1, 720, 8, 10), (1, 720, 4, 5), 
                (1, 36, 60, 80), (1, 36, 30, 40), (1, 36, 15, 20), (1, 36, 8, 10), (1, 36, 4, 5)]
outputs = [struct_out(o) for o in outputs_shape]

# run inference 
so_exe.model(image_arr.ctypes.data_as(ctypes.c_void_p), outputs[0], outputs[1], outputs[2], 
    outputs[3], outputs[4], outputs[5], outputs[6], outputs[7], outputs[8], outputs[9])
```

## 其他
- 本用例中使用的模型来自[onnx/models](https://github.com/onnx/models/tree/main/vision/object_detection_segmentation/retinanet)