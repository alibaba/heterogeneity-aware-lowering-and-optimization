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
import os
import argparse
import csv

build_dir = os.environ.get('HALO_BUILD_DIR')
if not os.path.exists(build_dir):
    print("can't find build path:({})".format(build_dir))
    exit(1)

src_dir = os.environ.get('HALO_SRC_DIR')
if not os.path.exists(src_dir):
    print("can't find build path:({})".format(src_dir))
    exit(1)

halo_bin = build_dir + '/bin/halo'
odla_inc = src_dir + '/ODLA/include'
odla_lib = build_dir + '/lib'
onnx_path = odla_lib + '/parser/onnx'
test_path = src_dir + '/tests/unittests'
test_include_path = test_path
tmp_dir = test_path + '/tmp'


# download onnx op data
def download_data(localdir):
    if not os.path.exists(localdir):
        os.system('git clone --depth 1 https://github.com/onnx/onnx.git')
        os.system('cp -r onnx/onnx/backend/test/data/node ' + localdir)
        os.system('rm -rf onnx')

# halo compile to generate cpp
def compile(case_path):
    halo_flags = []
    halo_flags.append('-target')
    halo_flags.append('cxx')
    halo_flags.append('--check-model')
    halo_flags.append('--emit-inference-func-sig')
    halo_flags.append('-batch-size=1')
    halo_flags_str = ' '
    halo_flags_str = halo_flags_str.join(halo_flags)

    case_name = os.path.basename(case_path)
    tmp_case_path = tmp_dir + '/' + case_name
    cxx_file = tmp_case_path + '.cc'
    obj_file = tmp_case_path + '.o'
    model_file = case_path + '/model.onnx'
    
    os.system(halo_bin + ' ' + model_file + ' ' 
              + halo_flags_str + ' -o ' + cxx_file)
    os.system('g++ ' + cxx_file + ' -c -fPIC -o ' 
              + obj_file + ' -I' + odla_inc)
    return obj_file

# g++ compile & link to generate test execute file
def link(case_path, obj_file, device, flags):
    include_path = []
    include_path.append('-I' + odla_inc)
    include_path.append('-I' + test_include_path)
    include_path.append('-I' + onnx_path)
    include_path_str = ' '
    include_path_str = include_path_str.join(include_path)

    lib_path = []
    lib_path.append('-L/usr/local/lib/ -lprotobuf')
    lib_path.append('-L'+ odla_lib + ' -lodla_' + device)
    lib_path_str = ' '
    lib_path_str = lib_path_str.join(lib_path)
    
    case_name = os.path.basename(case_path)
    tmp_case_path = tmp_dir + '/' + case_name
    exe_file = tmp_case_path + '_' + device + '.exe'
    bin_file = tmp_case_path + '.bin'
    test_file = tmp_case_path + '.main.cc'
    onnx_pb_cc_file = onnx_path + '/onnx.pb.cc'

    os.system('g++ '+ flags + ' ' + test_file + ' ' + onnx_pb_cc_file 
              + ' ' + obj_file + ' ' + bin_file + ' '
              + include_path_str + ' ' + lib_path_str 
              + ' -o ' + exe_file + ' -Wno-deprecated-declarations')
    #if ret_code != 0:
        #exit(1)
    return exe_file

# single case test
def single_test(test_case, device, error_thr, enable_timeperf, data_path):
    print('----------Test: [' + test_case + ':' + device + ']------')
    case_path = os.path.join(data_path, test_case)
    flags = ''
    if enable_timeperf == 'TRUE':
        flags = 'DTIME_PERF'
    obj_file = compile(case_path)
    exe_file = link(case_path, obj_file, device, flags)
    ld_libpath = os.environ.get('LD_LIBRARY_PATH')
    if odla_lib not in ld_libpath.split(':'):
        os.environ['LD_LIBRARY_PATH'] = ld_libpath + ':' + odla_lib
    ret_code = os.system(exe_file + ' ' + str(error_thr) 
                         + ' 0 ' + device + ' ' + case_path)
    if ret_code != 0:
        exit(1)
    
#all cases test    
def all_test(args, data_path):
    list_cases = os.listdir(data_path)
    devices = []
    devices.append('tensorrt')
    devices.append('dnnl')
    devices.append('eigen')
    devices.append('xnnpack')

    ld_libpath = os.environ.get('LD_LIBRARY_PATH')
    if odla_lib not in ld_libpath.split(':'):
        os.environ['LD_LIBRARY_PATH'] = ld_libpath + ':' + odla_lib
        print(os.environ['LD_LIBRARY_PATH'])

    flags = ''
    if args.enable_time_perf:
        flags = 'DTIME_PERF'

    for test_case in list_cases:
        print('----------Test: [' + test_case + ':' + str(devices) + ']------')
        error_thr = args.error_threshold
        case_path = os.path.join(data_path, test_case)
        obj_file = compile(case_path)
        for device in devices:
            exe_file = link(case_path, obj_file, device, flags)
            os.system(exe_file + ' ' + str(error_thr) 
                      + ' 0 ' + device + ' ' + case_path)

#test according to the config.csv
def list_test(data_path):
    config_file = test_path + '/config.csv'
    with open(config_file) as csvfile:
        reader = csv.reader(csvfile)
        for index,row in enumerate(reader):
            if index != 0 and row[5] == 'yes' and row[6] == 'PASS':
                single_test('test_'+row[0],
                row[1], row[2], row[3], data_path)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-mode', type=str, help='three test modes: single, all, list')
    parser.add_argument('--enable-time-perf', type=bool, default=False, help='whether it takes time test')
    parser.add_argument('--test-case', type=str, help='input test case name, only used in single case test')
    parser.add_argument('--error-threshold', type=float, help='comparison error with expected result')
    parser.add_argument('--device', type=str, help='ODLA supported backend: dnnl, eigen, tensorrt, xnnpack')
    return parser

if __name__ == "__main__":
    args = get_args().parse_args()

    data_path = test_path + '/data'
    download_data(data_path)

    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    if args.test_mode == 'single':
        single_test(args.test_case,
                    args.device,
                    args.error_threshold,
                    args.enable_time_perf, 
                    data_path)
    if args.test_mode == 'all':
        all_test(args, data_path)
    if args.test_mode == 'list':
        list_test(data_path)

    print("----------Test: Done---------------------")
