### Example of Using Inside Python <a name="example-of-using-inside-python"/>

HALO generated ODLA function can also be used inside Python.

Here we use [CaffeNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) as an example.

First, we compile the Caffe model into ODLA:
```bash
halo deploy.prototxt bvlc_reference_caffenet.caffemodel -target cxx -disable-broadcasting -entry-func-name=caffenet -batch-size=1 --input-shape=data:1x3x227x227 -o deploy.cc

g++ deploy.cc -c -fPIC -o deploy.o -I<halo_install_path>/include
```
Then, link it as a shared library using the TensorRT-based ODLA runtime library:
```bash
g++ -shared deploy.o deploy.bin -lodla_tensorrt -L <halo_install_path>/lib/ODLA -Wl,-rpath=<halo_install_path>/lib/ODLA -o /tmp/deploy.so
```

In a Python script, the CaffeNet inference can be invoked as:

```python
#...
c_lib = ctypes.CDLL('/tmp/deploy.so')
image = get_image_as_ndarray(path)
image = preprocess(image)
image = image.astype(ctypes.c_float)
ret = (ctypes.c_float * 1000)()
c_lib.caffenet(ctypes.c_void_p(image.ctypes.data), ret)
ret = np.array(ret)
ind = ret.argsort()[-3:][::-1]
#...
```

CaffeNet example can be found [here](models/vision/classification/caffenet).

