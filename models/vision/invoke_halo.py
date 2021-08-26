#!/usr/bin/env python3

"""
Copyright (C) 2019-2020 Alibaba Group Holding Limited.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
=============================================================================
"""

import subprocess
from PIL import Image
import os
import glob
import sys
import ctypes
import numpy as np
import shutil
import argparse
import time
import random
# Check halo
random.seed(10)
halo_exe = os.environ.get('HALO_BIN')
halo_include = os.environ.get('ODLA_INC')
halo_lib = os.environ.get('ODLA_LIB')
halo_tmp = os.environ.get('TEST_TEMP_DIR')
# CUDA_VISIBLE_DEVICES=2,3
os.environ["CUDA_VISIBLE_DEVICES"]="1" 
if not os.path.exists(halo_exe):
    print("Invalid HALO executable path:({})".format(halo_exe))
    exit(1)
if not os.path.exists(os.path.join(halo_include, 'ODLA', 'odla.h')):
    print("Invalid HALO include path:({})".format(halo_include))
    exit(1)
if not os.path.exists(os.path.join(halo_lib, 'libodla_dnnl.so')):
    print("Invalid HALO runtime library path:({})".format(halo_lib))
    exit(1)
if not os.access(halo_tmp, os.W_OK):
    print("Test temp path not writable:({})".format(halo_tmp))
    exit(1)

def verify(proc):
    print('\t' + ' '.join(proc.args))
    if proc.returncode != 0:
        print(proc.stderr)
        exit(proc.returncode)

def compile(model_path, halo_flags):
    output_stem = os.path.splitext(os.path.basename(model_path[0]))[0]
    cc_file = os.path.join(halo_tmp, output_stem + '.cc')
    obj_file = os.path.join(halo_tmp, output_stem + '.o')
    debug = '-g' if True else '-O2'

    print('\n[HALO] Compile {} ==> {}'.format(model_path, cc_file))

    verify(subprocess.run([halo_exe, *model_path,  *halo_flags,  '-o',
                           cc_file]))

    print('\n[Host Compiler] {} ==> {}'.format(cc_file, obj_file))
    verify(subprocess.run(
        ['g++', cc_file, debug, '-c', '-fPIC', '-o', obj_file, '-I' + halo_include, '-I' + '/usr/local/cuda-10.0/include']))
    return obj_file


def link(obj, as_shared, odla):
    bin = obj.split('.o')[0] + '.bin'
    exe = obj.split('.o')[0] + '.so'
    flags = '-shared' if as_shared else ''

    print('\n[Host Compiler] {} ==> {}'.format(obj, exe))
    # verify(subprocess.run(['g++', flags, obj, bin,
                        #    '-lodla_' + odla, '-L', halo_lib, '-Wl,-rpath=' + halo_lib, '-o', exe]))
    verify(subprocess.run(['g++', flags, obj, bin,
                           '-lodla_' + odla, '-L', halo_lib,'-lnvinfer', '-lcudart', '-Wl,-rpath=' + halo_lib, '-lcuda', '-o', exe]))
    print("daozhele")
    return exe


def compile_with_halo(model_path, func_name, odla_lib, layout_to, input_shapes, batch_size):
    halo_flags = []
    halo_flags.append('-target')
    halo_flags.append('cxx')
    halo_flags.append('-disable-broadcasting')
    halo_flags.append('-entry-func-name=' + func_name)
    halo_flags.append('-batch-size=' + str(batch_size))
    # halo_flags.append('-d')
    for s in input_shapes:
        halo_flags.append('--input-shape=' + s)
    if odla_lib == 'xnnpack' or odla_lib == 'eigen':
        halo_flags.append('-exec-mode=interpret')
    if odla_lib == 'xnnpack':
        halo_flags.append('-emit-value-id-as-int')
    # halo_flags.append('-fuse-conv-bias')
    if layout_to == 'nchw':
        halo_flags.append('-reorder-data-layout=channel-first')
        halo_flags.append('-remove-input-transpose')
    elif layout_to == 'nhwc':
        halo_flags.append('-reorder-data-layout=channel-last')
        halo_flags.append('-remove-input-transpose')
    obj = compile(model_path, halo_flags)
    return link(obj, True, odla_lib)


def run(exe):
    proc = subprocess.run([exe], stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE)
    print(proc)
    return proc.stdout


def get_image_as_ndarray(path, dst_h, dst_w):
    image = Image.open(path)
    image = image.resize((dst_h, dst_w))
    mode = image.mode
    image = np.array(image)  # type uint8
    if mode != 'RGB':
        print('Unsupported image mode {}'.format(mode))
        exit(1)
    return image


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', nargs='+', type=open)
    parser.add_argument('--label-file', type=open)
    parser.add_argument('--image-dir', type=str)
    parser.add_argument('--odla', type=str)
    parser.add_argument('--img-preprocess', type=str, default='')
    parser.add_argument('--convert-layout-to', type=str)
    parser.add_argument('--input_h', type=int, default=224)
    parser.add_argument('--input_w', type=int, default=224)
    parser.add_argument('--input-shape', type=str, nargs='+', default=[])
    parser.add_argument('--output_size', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--random_iter_nums', type=int, default=100)
    args = parser.parse_args()
    model_file = [m.name for m in args.model]
    image_path = args.image_dir
    if args.odla != "tensorrt":
        sys.exit(0)
    print('Using halo: {}'.format(halo_exe))

    model_lib = compile_with_halo(
        model_file, "model", args.odla, args.convert_layout_to, args.input_shape, args.batch_size)
    clsidx = []
    if args.label_file:
        clsidx = args.label_file.readlines()
    print('\n##### Start to test using {} #####'.format(model_lib))
    files = sorted(glob.glob(image_path + '/*.jpg'))

    if not files:
        print("No images found in " + image_path)
        exit(1)
    
    #暂时只用一张图片
    files = [image_path + '/dog.jpg']
    c_lib = ctypes.CDLL(model_lib)
    res_info = []
    time_with_preprocess = []
    time_without_preprocess = []
    ### preprocessing ###
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

    for path in random_select_data_iter(args.batch_size, files, args.random_iter_nums):
        start1 = time.time()
        images = []
        batch_size = len(path)       
        for image in path:
            image = get_image_as_ndarray(image, args.input_h, args.input_w)
            image = image.transpose([2, 0, 1])  # TO CHW
            pp = 'preprocess_image'
            if args.img_preprocess:
                pp = pp + '_' + args.img_preprocess
            pp_method = globals()[pp]
            image = pp_method(image)
            if args.convert_layout_to == 'nhwc':
                image = image.transpose([1, 2, 0])
            image = image.flatten().astype(ctypes.c_float)
            image = np.expand_dims(image, axis=0)
            images.append(image)
        if batch_size != args.batch_size:
            images.extend(np.zeros(image.shape)*(args.batch_size - args.batch_size))
        less_batch = args.batch_size - batch_size 
        images = np.array(images)
        ret = (ctypes.c_float * args.output_size * args.batch_size)()
        start2 = time.time()
        # 从这里输入的如果在model中
        c_lib.model(ctypes.c_void_p(images.ctypes.data), ret)
        # print(np.ctypeslib.as_array(images.ctypes.data)) 这是数据存储地址
        end2 = time.time()
        time_without_preprocess.append(end2 - start2)
        ret = np.array(ret)

        if clsidx:
            for i in range(len(path)):
                ind = ret[i].argsort()[-3:][::-1]
                # print(ind)
                # print(os.path.basename(path[i]), '==>', clsidx[ind[0]].strip('\n'))
                res_info.append(path[i].split('/')[-1] + ' ==> ' + clsidx[ind[0]])
        else:
            print(ind)
            res_info.append(ind)
        # print(res_info)
        end1 = time.time()
        time_with_preprocess.append(end1 - start1)
        # print(ret)

    # for i, path in enumerate(files):
    #     start1 = time.time()
    #     image = get_image_as_ndarray(path, args.input_h, args.input_w)
    #     image = image.transpose([2, 0, 1])  # TO CHW
    #     pp = 'preprocess_image'
    #     if args.img_preprocess:
    #         pp = pp + '_' + args.img_preprocess
    #     pp_method = globals()[pp]
    #     image = pp_method(image)
    #     if args.convert_layout_to == 'nhwc':
    #         image = image.transpose([1, 2, 0])
    #     image = image.flatten().astype(ctypes.c_float)
    #     image = np.expand_dims(image, axis=0)
    #     image_2 = np.concatenate([image, image], axis=0)
    #     ret = (ctypes.c_float * args.output_size*2)()
    #     start2 = time.time()
    #     c_lib.model(ctypes.c_void_p(image_2.ctypes.data), ret)
    #     end2 = time.time()
    #     time_without_preprocess.append(end2 - start2)
    #     ret = np.array(ret)
    #     print(ret)
    #     ind = ret.argsort()[-3:][::-1]

    #     if clsidx:
    #         print(os.path.basename(path), '==>', clsidx[ind[0][0]].strip('\n'))
    #         res_info.append(path.split('/')[-1] + ' ==> ' + clsidx[ind[0][0]])
    #     else:
    #         print(ind)
    #         res_info.append(ind)
    #     end1 = time.time()
    #     time_with_preprocess.append(end1 - start1)
    #resfile = os.path.splitext(os.path.basename(model_file[0]))[0]
    #resfile = resfile + '_' + args.odla + '.txt'
    #resfile = os.path.join('/tmp', resfile)
    #with open(resfile, 'w') as f:
    #    f.write(''.join(str(i) for i in res_info))
    # print("======== Testing Done ========")
    # print(res_info)
    # print("The time of reference without preprocessing is {}, and the time with prerprocessing is {}".format(time_without_preprocess, time_with_preprocess))
    # print(time_without_preprocess)
    avg_image_time = sum(time_without_preprocess[2:])/(args.random_iter_nums*args.batch_size - args.batch_size*2)*1000
    print(args.model)
    avg_batch_time = sum(time_without_preprocess[2:])/(args.random_iter_nums - 2)*1000
    print(avg_batch_time)
    print(avg_image_time)
    # print(time_without_preprocess[1:])
    std_batch_time = np.std(np.array(time_without_preprocess[2:]))*1000
    print(std_batch_time)
    std_image_time = np.std(np.array([i/args.batch_size for i in time_without_preprocess[2:]]))*1000
    print(std_image_time)
    # print([i*1000 for i in time_without_preprocess])
    # with open("vision/Plot/inceptionv3_odla_before_avg_std_result.txt", "a+") as f:
    #     f.write(str([avg_batch_time, std_batch_time, avg_image_time, std_image_time]) + '\n')

    
    # with open("odla_tensorrt_times.txt", "r") as f:
    #     lines = f.readlines()
    #     time_per_iter = [float(line.split("\n")[0]) for line in lines]
    #     average_execute_time = sum(time_per_iter[-1*args.random_iter_nums + 1:])/(args.random_iter_nums*args.batch_size - args.batch_size)
    #     print(average_execute_time)
    #     print(sum(time_per_iter[-1*args.random_iter_nums + 1:])/(args.random_iter_nums - 1))