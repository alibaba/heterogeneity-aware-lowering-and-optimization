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
import threading
import csv

build_dir = os.environ.get('HALO_BUILD_DIR')
if not os.path.exists(build_dir):
    print("can't find build path:({})".format(build_dir))
    exit(1)

src_dir = os.environ.get('HALO_SRC_DIR')
if not os.path.exists(src_dir):
    print("can't find build path:({})".format(src_dir))
    exit(1)

halo_bin = os.path.join(build_dir, 'bin/halo')
odla_inc = os.path.join(src_dir, 'ODLA/include')
odla_lib = os.path.join(build_dir, 'lib')
onnx_path = os.path.join(odla_lib, 'parser/onnx')
test_path = os.path.join(src_dir, 'tests/unittests')
lit_cases_path = os.path.join(test_path, 'lit_cases')
test_include_path = test_path
test_build_path = os.path.join(build_dir, 'tests/unittests')
tmp_dir = os.path.join(test_build_path, 'tmp')

ld_libpath = os.environ.get('LD_LIBRARY_PATH')
if odla_lib not in ld_libpath.split(':'):
    os.environ['LD_LIBRARY_PATH'] = ld_libpath + ':' + odla_lib


# download onnx op data
def download_data(localdir):
    if not os.path.exists(localdir):
        os.system('git clone -b rel-1.5.0 https://github.com/onnx/onnx.git')
        os.system('cp -r onnx/onnx/backend/test/data/node ' + localdir)
        os.system('rm -rf onnx')


def run_shell_cmd(args):
    proc = subprocess.run(args, stdout=subprocess.PIPE)
    # return  errcode, shell commands, stdout and stderr.
    return([proc.returncode, ' '.join(proc.args), proc.stdout, proc.stderr])

# halo compile to generate cpp
def compile(case_path, device):
    halo_flags = []
    halo_flags.append('-target')
    halo_flags.append('cxx')
    halo_flags.append('--check-model')
    halo_flags.append('--emit-inference-func-sig')
    if (device == 'xnnpack' or device == 'eigen'):
        halo_flags.append('--reorder-data-layout=channel-first')
    halo_flags.append('-batch-size=1')

    case_name = os.path.basename(case_path)
    tmp_case_path = os.path.join(tmp_dir, case_name)
    cxx_file = tmp_case_path + '.cc'
    obj_file = tmp_case_path + '.o'
    model_file = os.path.join(case_path, 'model.onnx')

    ret = run_shell_cmd([halo_bin, model_file,
                         *halo_flags, '-o', cxx_file])
    if ret[0]:
        return [None, *ret]
    ret = run_shell_cmd(['g++', cxx_file, '-c', '-g', '-fPIC', '-o',
                         obj_file, '-I', odla_inc])
    if ret[0]:
        return [None, *ret]
    return [obj_file, *ret]

# g++ compile & link to generate test execute file
def link(case_path, obj_file, device, flags):
    include_path = []
    include_path.append('-I' + odla_inc)
    include_path.append('-I' + test_include_path)
    include_path.append('-I' + onnx_path)

    lib_path = []
    lib_path.append('-L/usr/local/lib/')
    lib_path.append('-lprotobuf')
    lib_path.append('-L' + odla_lib)
    lib_path.append('-lodla_' + device)
    lib_path.append('-Wl,-rpath=' + odla_lib)

    case_name = os.path.basename(case_path)
    tmp_case_path = tmp_dir + '/' + case_name
    exe_file = tmp_case_path + '_' + device + '.exe'
    bin_file = tmp_case_path + '.bin'
    test_file = tmp_case_path + '.main.cc'
    onnx_pb_cc_file = onnx_path + '/onnx.pb.cc'

    ret = run_shell_cmd(['cp', test_file + '.in', test_file])
    if ret[0]:
        return [None, *ret]

    ret = run_shell_cmd(
        ['g++', *flags, test_file,  onnx_pb_cc_file,
         obj_file, bin_file, *include_path, *lib_path,
         '-o', exe_file, '-Wno-deprecated-declarations'])
    if ret[0]:
        return [None, *ret]
    return [exe_file, *ret]

results = []

def print_results():
    passed = list(filter(lambda x: x[2], results))
    failed = list(filter(lambda x: not x[2], results))
    print('Total tests: {0}'.format(len(results)))
    print('Passed: {0}'.format(len(passed)))
    print('Failed: {0}'.format(len(failed)))

# single case test
def single_test(test_case, device, error_thr, enable_timeperf, data_path):
    print('----------Test: [' + test_case + ':' + device + ']------')
    case_path = os.path.join(data_path, test_case)
    flags = ['-g']
    if enable_timeperf == 'TRUE':
        flags.append('-DTIME_PERF')
    ret = compile(case_path, device)
    obj_file = ret[0]
    if not obj_file:
        results.append([test_case, device, False, ret[-1:]])
        return ret
    ret = link(case_path, obj_file, device, flags)
    exe_file = ret[0]
    if not exe_file:
        results.append([test_case, device, False, ret[-1:]])
        return ret
    ret = run_shell_cmd([exe_file, error_thr, '0', device, case_path])
    if ret[0]:
        results.append([test_case, device, False, ret])
    else:
        if ret[-2].decode("utf-8") == 'Result Pass\n':
            results.append([test_case, device, True, ret])
        else:
            results.append([test_case, device, False, ret])
    return ret

def get_copyright(case_name, device):
    cright = ['//===-']
    cright.append(case_name + '_' + device + '.cc')
    cright.append('-----------------------------------------------------------===//\n')
    cright.append('//\n')
    cright.append('// Copyright (C) 2019-2020 Alibaba Group Holding Limited.\n')
    cright.append('//\n')
    cright.append('// Licensed under the Apache License, Version 2.0 (the "License");\n')
    cright.append('// you may not use this file except in compliance with the License.\n')
    cright.append('// You may obtain a copy of the License at\n')
    cright.append('//\n')
    cright.append('//   http://www.apache.org/licenses/LICENSE-2.0\n')
    cright.append('//\n')
    cright.append('// Unless required by applicable law or agreed to in writing, software\n')
    cright.append('// distributed under the License is distributed on an "AS IS" BASIS,\n')
    cright.append('// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n')
    cright.append('// See the License for the specific language governing permissions and\n')
    cright.append('// limitations under the License.\n')
    cright.append('// =============================================================================\n')
    cright.append('\n')
    return ''.join(cright)

def add_single_litcase(lit_path, test_case, device, error_threshold, data_path):

    ret = single_test(test_case, device, error_threshold, 'FALSE', data_path)

    print('----------Add lit case: [' + test_case + ':' + device + ']------')
    if not os.path.exists(lit_path):
        os.mkdir(lit_path)
    lit_test_file = os.path.join(lit_path, test_case + '_' + device + '.cc')
    halo_compile = ['// RUN: %halo_compiler ']
    halo_compile.append('-target cxx ')
    halo_compile.append('-batch-size 1 ')
    halo_compile.append('%halo_compile_flags ')
    halo_compile.append('%data_path/')
    halo_compile.append(test_case)
    halo_compile.append('/model.onnx ')
    halo_compile.append('-o %t.cc\n')
    halo_compile_str = ''.join(halo_compile)

    gencxx_compile = ['// RUN: %cxx ']
    gencxx_compile.append('-c -fPIC -o ')
    gencxx_compile.append('%t.o %t.cc ')
    gencxx_compile.append('-I%odla_path/include\n')
    gencxx_compile_str = ''.join(gencxx_compile)

    testmain_compile = ['// RUN: %cxx -g ']
    testmain_compile.append('%s ')
    testmain_compile.append('%onnx_path')
    testmain_compile.append('/onnx.pb.cc ')
    testmain_compile.append('%t.o ')
    testmain_compile.append('%t.bin ')
    testmain_compile.append('-I%T ')
    testmain_compile.append('-I%odla_path/include ')
    testmain_compile.append('-I%unittests_path ')
    testmain_compile.append('-I%onnx_path ')
    testmain_compile.append('-L/usr/local/lib/ ')
    testmain_compile.append('-lprotobuf ')
    testmain_compile.append('%odla_link ')
    testmain_compile.append('-lodla_' + device)
    testmain_compile.append(' -o %t_' + device + '.exe')
    testmain_compile.append(' -Wno-deprecated-declarations\n')
    testmain_compile_str = ''.join(testmain_compile)

    exec_check = ['// RUN: %t_' + device + '.exe ']
    exec_check.append(error_threshold)
    exec_check.append(' 0 ')
    exec_check.append(device)
    exec_check.append(' %data_path/')
    exec_check.append(test_case)
    exec_check.append(' | FileCheck %s\n')
    exec_check.append('// CHECK: Result Pass\n')
    exec_check_str = ''.join(exec_check)

    lit_header = [get_copyright(test_case, device)]
    lit_header.append('// clang-format off\n')
    lit_header.append('// Testing CXX Code Gen using ODLA API on ')
    lit_header.append(device)
    lit_header.append('\n')
    lit_header.append(halo_compile_str)
    lit_header.append(gencxx_compile_str)
    lit_header.append(testmain_compile_str)
    lit_header.append(exec_check_str)
    lit_header.append('// clang-format on\n')

    lit_body = '\n#include "' + test_case + '_' + device +'.cc.tmp.main.cc.in"\n'
    lit_excepted_fail = "// XFAIL: *\n"

    if ret[-2].decode("utf-8") == 'Result Pass\n':
        lit_excepted_fail = "\n"

    with open(lit_test_file, 'w') as litcasefile:
        litcasefile.write(''.join(lit_header))
        litcasefile.write(lit_excepted_fail)
        litcasefile.write(lit_body)

def add_all_litcases(data_path, error_threshold):
    if not os.path.exists(test_build_path):
        os.mkdir(test_build_path)
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    if not os.path.exists(lit_cases_path):
        os.mkdir(lit_cases_path)

    threads = []

    list_cases = os.listdir(data_path)
    devices = ['tensorrt']
    devices.append('dnnl')
    devices.append('eigen')
    devices.append('xnnpack')

    for test_case in list_cases:
        print('----------Add lit cases: [' + test_case + ':' + str(devices) + ']------')
        case_path = os.path.join(data_path, test_case)
        for device in devices:
            # force single thread to execute for tensorrt
            if device == 'tensorrt':
                for t in threads:
                    t.join()
            lit_case_path = os.path.join(lit_cases_path, 'test_' + device)
            t = threading.Thread(target=add_single_litcase, args=[lit_case_path,
                                        test_case, device, error_threshold, data_path])
            threads.append(t)
            t.start()
            if device == 'tensorrt':
                t.join()
    for t in threads:
        t.join()
    print_results()

if __name__ == "__main__":
    if not os.path.exists(test_build_path):
        os.mkdir(test_build_path)
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    if not os.path.exists(lit_cases_path):
        os.mkdir(lit_cases_path)
    data_path = os.path.join(build_dir, 'tests/unittests/data')
    download_data(data_path)
    add_all_litcases(data_path, '0.0001')

    print("----------ADD lit cases: Done---------------------")
