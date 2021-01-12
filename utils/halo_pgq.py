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
import json

use_pillow = False
try:
    import cv2
except ImportError:
    print("OpenCV not found. Run pip3 install scikit-build opencv-python-headless")
    printf("Use Pillow")
    use_pillow = True
    pass


def verify(proc):
    if verbose:
        print(proc.args)
    if proc.returncode != 0:
        print(proc.args)
        print(proc.stderr)
        exit(proc.returncode)


def compile(model_files, halo_flags, is_prof):
    model_file = model_files[0]
    file_stem = os.path.basename(os.path.splitext(model_file)[0])
    output_dir = '/tmp' if is_prof else os.path.dirname(model_file)
    output_stem = file_stem + ('_prof' if is_prof else '')
    cc_file = os.path.join(output_dir, output_stem + '.c')
    bin_file = os.path.join(output_dir, output_stem + '.bin')
    obj_file = os.path.join(output_dir, output_stem + '.o')
    debug = '-g' if True else '-O2'
    verify(subprocess.run([halo_exe, *model_files,  *halo_flags,  '-o',
                           cc_file]))
    if not is_prof:
        return (cc_file, bin_file)
    verify(subprocess.run(
        ['gcc', cc_file, debug, '-c', '-fPIC', '-o', obj_file, '-I/halo/include']))
    return (obj_file, bin_file)


def link(obj, bin, as_shared):
    exe = obj.split('.o')[0] + '.so'
    flags = '-shared' if as_shared else ''
    verify(subprocess.run(['g++', flags, obj, bin,
                           '-lodla_profiler', '-L', halo_lib, '-Wl,-rpath=' + halo_lib, '-o', exe]))
    return exe


def compile_with_halo(model_files, func_name, is_prof, skip_weights_quan, prof_result_file=''):
    halo_flags = []
    needs_relayout = False
    if len(model_files) == 1 and os.path.splitext(model_files[0])[1] == '.pb':
        needs_relayout = True
    halo_flags.append('-target')
    halo_flags.append('cc')
    halo_flags.append('-disable-broadcasting')
    halo_flags.append('-entry-func-name=' + func_name)
    halo_flags.append('-fuse-conv-bias')
    halo_flags.append('-fuse-matmul-bias')
    if needs_relayout:
        halo_flags.append('-remove-input-transpose')
        halo_flags.append('-reorder-data-layout=channel-first')
    if is_prof:
        halo_flags.append('-exec-mode=interpret')
    else:
        if not skip_weights_quan:
            halo_flags.append('-quantize-weights=quint8')
        halo_flags.append('-emit-data-as-c')
        halo_flags.append('-pgq-file=' + prof_result_file)

    obj, bin = compile(model_files, halo_flags, is_prof)
    return link(obj, bin, True) if is_prof else (obj, bin)


def run(exe):
    proc = subprocess.run([exe], stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE)
    print(proc)
    return proc.stdout


def get_image_ndarray(path):
    if use_pillow:
        image = Image.open(path)
        image = image.resize((112, 112))
        mode = image.mode
        image = np.array(image)  # type uint8
        if mode == 'L':
            image = np.expand_dims(image, axis=2)
            image = image.repeat(3, axis=2)
        elif mode == 'RGB':
            pass
        else:
            print('image mode has to be either L or RGB, but got {}'.format(mode))
    else:
        image = cv2.imread(path)
        # image = cv2.resize()
    return image


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', nargs='+', dest='model', required=True,
                        type=open, help='model files')
    parser.add_argument('-i', '--image-dir', required=True,
                        type=str, help='image directory')
    parser.add_argument('-s', '--skip-weights-prof',
                        dest='skip_weights_prof', action='store_true', default=False, help='No profiling for weights')
    parser.add_argument('-c', '--chs-prof', dest='chs_prof',
                        action='store_true', default=False, help='Enable channel-wise profiling')
    parser.add_argument('-a', '--chs-axis', dest='chs_axis', type=int,
                        help='axis of channel (e.g.: 1 for NCHW, 3 for NHWC)', default=1)
    parser.add_argument('-v', '--verbose', dest='verbose',
                        action='store_true', default=False)
    args = parser.parse_args()

    model_files = [x.name for x in args.model]
    image_path = args.image_dir

    global halo_exe
    global halo_home
    global halo_lib
    global verbose

    verbose = args.verbose

    # Check halo
    halo_exe = shutil.which('halo')
    if not halo_exe:
        print('halo not found. Please add it to PATH')
        exit(1)

    print("Halo: " + halo_exe)
    halo_home = os.path.dirname(halo_exe) + '/..'
    halo_lib = halo_home + '/lib'

    print("\n##### Compile model (for profiling) #####")
    model_lib = compile_with_halo(
        model_files, 'model', True, args.skip_weights_prof)
    print("Generated ", model_lib)

    print("\n##### Start to profile #####")

    files = glob.glob(image_path + '/*.jpg')

    if not files:
        print("No images found in " + image_path)
        exit(1)
    else:
        print(str(len(files)) + " images found")
    print(files)

    c_lib = ctypes.CDLL(model_lib)
    prof_file = os.path.splitext(model_lib)[0] + '.prof'
    prof_file_buf = prof_file.encode('utf-8')
    prof_file_p = ctypes.c_char_p(prof_file_buf)
    c_lib.StartProfiling(prof_file_p)
    if args.chs_prof:
        c_lib.EnableChannelWiseProf()
        c_lib.SetChannelAxis(args.chs_axis)  # NCHW
    if args.skip_weights_prof:
        c_lib.SkipWeightsProfiling(1)

    ### preprocessing ###
    for i, path in enumerate(files):
        image = get_image_ndarray(path)
        image = image.transpose([2, 0, 1]).flatten()  # TO CHW
        image = image.astype(ctypes.c_float)
        # np.savetxt('image_' + str(i) + '.inc', image.flatten(),
        #           delimiter=',', newline=',\n')
        ret0 = (ctypes.c_float * 28224 * 10)()
        ret1 = (ctypes.c_float * 28224 * 2)()
        ret2 = (ctypes.c_float * 28224 * 4)()

        print('Profiling {} of {}: {}'.format(i, len(files), path))
        tag = path.encode('utf-8')
        c_lib.StartOneRun(ctypes.c_char_p(tag))
        c_lib.model(ctypes.c_void_p(image.ctypes.data), ret0, ret1, ret2)
        c_lib.StopOneRun()
        # ret = np.array(ret)
        # ind = ret.argsort()[-3:][::-1]
        # print(ind, ret[ind])
    c_lib.StopProfiling()

    # Consolidate profiling results
    with open(prof_file, 'r') as f:
        prof_data = json.load(f)
    prof_data = prof_data['ProfilingResults']
    consolidated = {}
    for run, run_data in prof_data.items():
        for layer_name, prof_vals in run_data.items():
            # Validate
            chs = prof_vals['channels']
            if chs != len(prof_vals['channels_max']) or chs != len(prof_vals['channels_min']):
                print("Skip invalid record")
                continue

            if not layer_name in consolidated:
                consolidated[layer_name] = prof_vals
            else:
                v = consolidated[layer_name]['min_value']
                consolidated[layer_name]['min_value'] = min(
                    v, prof_vals['min_value'])
                v = consolidated[layer_name]['max_value']
                consolidated[layer_name]['max_value'] = max(
                    v, prof_vals['max_value'])
                op_type = prof_vals['op_type']
                if consolidated[layer_name]['channels'] != chs or \
                        consolidated[layer_name]['op_type'] != op_type:
                    print("Skip invalid record")
                    continue
                v = consolidated[layer_name]['channels_min']
                consolidated[layer_name]['channels_min'] = min(
                    v, prof_vals['channels_min'])
                v = consolidated[layer_name]['channels_max']
                consolidated[layer_name]['channels_max'] = max(
                    v, prof_vals['channels_max'])

    # Compute scale and zp based on 8-bit quant rule
    for data in consolidated.values():
        valid_range = 255
        scale = (data['max_value'] - data['min_value']) / valid_range
        if scale == 0:
            scale = abs(data['max_value'])
        if scale == 0:
            scale = 1
        zp = min(255, max(0, round(0 - data['min_value'] / scale)))
        data['scale'] = scale
        data['zp'] = round(zp)

    output_file = os.path.splitext(model_lib)[0] + '.json'

    # Output json
    with open(output_file, 'w') as fp:
        json.dump(consolidated, fp)
    print('Consolidated JSON file:{}'.format(output_file))

    # Output C code, assumes quant
    output_file = os.path.splitext(model_lib)[0] + '.c'
    with open(output_file, 'w') as cf:
        cf.write('#include <ODLA/ops/odla_ops_quantization.h>\n')
        cf.write('const int quant_infos_size = {};\n'.format(
            len(consolidated.items())))
        cf.write('const odla_value_quant_info quant_infos[] = {\n')
        fmt = '  {{.value_id = (const odla_value_id) "{}", .ch_idx = {}, .scale = {}, .offset = {}, .min = {}, .max = {}}},\n'
        for name, data in consolidated.items():
            chs = data['channels']
            cf.write(fmt.format(
                name, -chs, data['scale'], data['zp'], data['min_value'], data['max_value']))
            for ch in range(0, chs):
                cf.write(fmt.format(
                    name, ch, 0, 0, data['channels_min'][ch], data['channels_max'][ch]))
        cf.write('};')
    print('ODLA Quantization info file:{}'.format(output_file))

    # Output csv
    output_file = os.path.splitext(model_lib)[0] + '.csv'
    with open(output_file, 'w') as fp:
        for name, data in consolidated.items():
            fp.write('{},{},{},{},{}\n'.format(
                name, data['min_value'], data['max_value'], data['scale'], data['zp']))
    print('Quantization csv file:{}'.format(output_file))

    # Final compile
    print("\n##### Compile model (with profiled info) #####")
    final_cc, final_bin = compile_with_halo(
        model_files, 'model', False, args.skip_weights_prof, output_file)
    print('Final outputs: {}, {}'.format(final_cc, final_bin))
