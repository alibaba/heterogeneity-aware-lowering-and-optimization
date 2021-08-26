#!/usr/bin/env python3

import os
import sys
import ctypes
import random
import time 
import warnings

import numpy as np
import pycuda.autoinit
import tensorrt as trt
import pycuda.driver as cuda 
import argparse
import glob
from PIL import Image
import common
import shutil
random.seed(10)
warnings.filterwarnings('ignore')
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt_runtime = trt.Runtime(TRT_LOGGER)
os.environ["CUDA_VISIBLE_DEVICES"]="2" 

def get_engine(onnx_file_path, engine_file_path):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""
    def build_engine():
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        builder = trt.Builder(TRT_LOGGER)
        # if max_batch_size > 1:
        builder.max_batch_size = 1

        # builder.batchsize = batch_size
        network = builder.create_network(common.EXPLICIT_BATCH)
        config = builder.create_builder_config()
        # print(config)
        config.max_workspace_size = 1 << 28
        parser = trt.OnnxParser(network, TRT_LOGGER)

        # Parse model file
        print('Loading ONNX file from path {}...'.format(onnx_file_path))
        with open(onnx_file_path, 'rb') as model:
            # parser.parse(model.read())
            print('Beginning ONNX file parsing')
            if not parser.parse(model.read()):
                print('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None

        # network.get_input(0).shape = shape
        print('Completed parsing of ONNX file')
        engine = builder.build_cuda_engine(network)

        with open(engine_file_path, "wb") as f:
            f.write(engine.serialize())
        return engine
     
    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine()

def save_engine(engine, file_name):
    buf = engine.serialize()
    with open(file_name, 'wb') as f:
        f.write(buf)

def load_engine(trt_runtime, engine_path):
    with open(engine_path,'rb') as f:
        engine_data = f.read()
    engine = trt_runtime.deserialize_cuda_engine(engine_data)
    return engine

def preprocess_image(img_data):
    mean_vec = np.array([0.485, 0.456, 0.406])
    stddev_vec = np.array([0.229, 0.224, 0.225])
    norm_img_data = np.zeros(img_data.shape).astype('float32')
    for i in range(img_data.shape[0]):
        norm_img_data[i, :, :] = (
            img_data[i, :, :] / 255.0 - mean_vec[i]) / stddev_vec[i]
    return norm_img_data

def preprocess_image_minus_128(img_data):
    #mean_vec = np.array([122.7717, 115.9465, 102.9801])
    mean_vec = np.array([128.0, 128.0, 128.0])
    norm_img_data = np.zeros(img_data.shape).astype('float32')
    for i in range(img_data.shape[0]):
        norm_img_data[i, :, :] = img_data[i, :, :] - mean_vec[i]
    return norm_img_data

# def load_normalized_test_case(test_image, pagelocked_buffer):
#     # Converts the input image to a CHW Numpy array
#     def normalize_image(image):
#         # Resize, antialias and transpose the image to CHW.
#         c, h, w = ModelData.INPUT_SHAPE
#         image_arr = image.astype(trt.nptype(ModelData.DTYPE)).ravel()
#        # This particular ResNet50 model requires some preprocessing, specifically, mean normalization.
#         return (image_arr / 255.0 - 0.45) / 0.225

#     # Normalize the image and copy to pagelocked memory.
#     np.copyto(pagelocked_buffer, normalize_image(image))
#     return test_image

def get_image_as_ndarray(path, dst_h, dst_w):
    image = Image.open(path)
    image = image.resize((dst_h, dst_w))
    mode = image.mode
    image = np.array(image)  # type uint8
    if mode != 'RGB':
        print('Unsupported image mode {}'.format(mode))
        exit(1)
    return image
    
def data_iter(batch_size, features):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        j = np.array(indices[i: min(batch_size+i, num_examples)])
        result = []
        for idx in j:
            result.append(features[idx])
        yield result

def random_select_data_iter(batch_size, features, iter_nums):
    for i in range(iter_nums):
        result = random.choices(features, k=batch_size)
        yield result 

# # 生成mode_data.h 
# def generate_hfile(args):
#     model_name = args.model.split("/")[-1].split(".")[0]
#     input_size = "1*3*"+str(args.input_h)+"*"+str(args.input_w)
#     output_size = str(args.output_size)
#     hfile_name = model_name + '_data.h'
#     with open(hfile_name, "w+"):
#         f.write('static const float input_')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--label-file', type=open, default='./classification/1000_labels.txt')
    parser.add_argument('--image_dir', type=str, default='/models/vision/test_images')
    # parser.add_argument('--odla', type=str)
    parser.add_argument('--img-preprocess', type=str, default='')
    parser.add_argument('--convert-layout-to', type=str)
    parser.add_argument('--input_h', type=int, default=224)
    parser.add_argument('--input_w', type=int, default=224)
    parser.add_argument('--input-shape', type=str, nargs='+', default=[])
    parser.add_argument('--output_size', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--random_iter_nums', type=int, default=100)
    args = parser.parse_args()
    # model_file = [m.name for m in args.model]
    # image_path = args.image_dir

    # engine = build(args.model, [1, 3, 224, 224])
    # context = engine.create_execution_context()
    class ModelData(object):
        if args.convert_layout_to == "nhwc":
            INPUT_SHAPE = [args.batch_size, args.input_h, args.input_w, 3]
        else:
            INPUT_SHAPE = [args.batch_size, 3, args.input_h, args.input_w]
        DTYPE = trt.float32
    input_shape = ModelData.INPUT_SHAPE
    model = args.model
    if args.batch_size > 1:
        model = model.split(".")[-2] + "_bs" + str(args.batch_size) + ".onnx"
    
    engine_path = args.model.split("/")[-1].split(".")[0]
    model_name = engine_path
    engine_file_path = './tensorrt_files/'+engine_path+'_batch'+str(args.batch_size)+'.trt'

    engine = get_engine(model, engine_file_path)
    
    context = engine.create_execution_context()
    input_output_file = './input_output/'+model_name

    isExists = os.path.exists(input_output_file)
    if not isExists:
        os.makedirs(input_output_file)

    image_path = args.image_dir
    INPUT_SHAPE = ModelData.INPUT_SHAPE
    DTYPE = trt.float32
    clsidx = []
    if args.label_file:
        clsidx = args.label_file.readlines()

    files = sorted(glob.glob(image_path + '/*.jpg'))

    files = [image_path + '/dog.jpg']
    if not files:
        print("No images found in " + image_path)
        exit(1)

    inputs, outputs, bindings, stream = common.allocate_buffers(engine)
    test_list = []
    result = []
    # 填充model_data.h 内容
    hfile_context = ""
    label_result = []
    # for i, path in enumerate(files):
        # input_filename = input_output_file + '/input_data_' + str(i) + '.txt'
        # if args.convert_layout_to == 'nhwc':
        #     str_input_size = "[1 * "+str(args.input_h)+" * "+str(args.input_w)+" * 3]"
        # else:
        #     str_input_size = "[1 * 3 * "+str(args.input_h)+" * "+str(args.input_w)+"]"
        # input_class_context = "static const float input_"+str(i)+str_input_size+"= { \n#include \"input_data_"+str(i)+".txt\"\n};\n\n"
        # hfile_context = hfile_context + input_class_context
    for paths in random_select_data_iter(args.batch_size, files, args.random_iter_nums):
        images = []
        images_batch = []
        # isExists = os.path.exists(input_filename)
        # if isExists:
        #     image = np.loadtxt(input_filename)
        #     with open(input_filename, 'r') as f:
        #         data = f.read()
        #         data = data.split('\n')
        #         image = map(float, data)
        #         print(image)
        for i, path in enumerate(paths):
            image = get_image_as_ndarray(path, args.input_h, args.input_w)
            image = image.transpose([2,0,1])
            pp = 'preprocess_image'
            if args.img_preprocess:
                pp = pp + '_' + args.img_preprocess
            pp_method = globals()[pp]
            image = pp_method(image)
            if args.convert_layout_to == 'nhwc':
                image = image.transpose([1, 2, 0])
            images.append(image)
            image = image.flatten().astype(ctypes.c_float)
        images = np.array(images)
        if args.batch_size == 1:
            images = images[0]
        images = images.flatten().astype(ctypes.c_float)
        pagelocked_buffer = inputs[0].host
        # np.copyto(pagelocked_buffer, images)
        np.copyto(pagelocked_buffer, images)
        # image = load_normalized_test_case(image, inputs[0].host)
        start = time.time()
        trt_output, execute_cost_time = common.do_inference_v3(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)#, batch_size=args.batch_size)
        end = time.time()
        # output_filename = input_output_file + '/output_data_' + str(i) + '.txt'
        # output_class_context = "static const float output_ref_"+str(i)+"["+str(args.output_size)+"] = { \n#include \"output_data_"+str(i)+".txt\"\n};\n\n"
        # hfile_context = hfile_context + output_class_context
        # hfile_context += "static float output_"+str(i)+"["+str(args.output_size)+"]; \n\n"
        # with open(output_filename, "w+") as f:
        #     for j in trt_output[0]:
        #         f.write(str(j)+'\n')
        cost_time = end - start 
        result.append(cost_time)
        pred = []
        for i in range(args.batch_size):
            # print(trt_output[0][i*args.output_size: (i+1)*args.output_size])
            pred.append(clsidx[np.argmax(trt_output[0][i*args.output_size: (i+1)*args.output_size])])
        # pred = clsidx[np.argmax(trt_output[0])]
        label_result.append(pred)
    # print(label_result)
    # print(result)
    print(engine_file_path)
    # print(result)
    avg_batch_time = sum(result[1:])/(args.random_iter_nums - 1)*1000
    print(avg_batch_time)
    avg_image_time = sum(result[1:])/(args.random_iter_nums*args.batch_size - args.batch_size)*1000
    print(avg_image_time)
    std_batch_time = np.std(np.array(result[1:]))*1000
    print(std_batch_time)
    std_image_time = np.std(np.array([i/args.batch_size for i in result[1:]]))*1000
    print(std_image_time)
    
    # with open("Plot/inceptionv3_trt_avg_std_result.txt", "a+") as f:
    #     f.write(str([avg_batch_time, std_batch_time, avg_image_time, std_image_time]) + '\n')

    # print(label_result)
    # with open("result.txt","a+") as f:
    #     f.write(engine_path+':'+" ".join(result)+"\n")
    # # print(result)
    # with open(input_output_file +"/"+engine_path + '_data.h', "w+") as f:
    #     f.write(hfile_context)
    # pred = np.argmax(trt_outputs[0])
    # print(pred)
    # print(clsidx[pred])