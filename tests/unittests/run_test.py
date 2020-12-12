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
import contextlib

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
test_include_path = test_path
tmp_dir = os.path.join(build_dir, 'tests/unittests/tmp')

ld_libpath = os.environ.get('LD_LIBRARY_PATH')
if odla_lib not in ld_libpath.split(':'):
    os.environ['LD_LIBRARY_PATH'] = ld_libpath + ':' + odla_lib


# download onnx op data
def download_data(localdir):
    if not os.path.exists(localdir):
        os.system('git clone --depth 1 https://github.com/onnx/onnx.git')
        os.system('cp -r onnx/onnx/backend/test/data/node ' + localdir)
        os.system('rm -rf onnx')


def run_shell_cmd(args):
    proc = subprocess.run(args)
    # return  errcode, shell commands, stdout and stderr.
    return([proc.returncode, ' '.join(proc.args), proc.stdout, proc.stderr])

# halo compile to generate cpp


def compile(case_path):
    halo_flags = []
    halo_flags.append('-target')
    halo_flags.append('cxx')
    halo_flags.append('--check-model')
    halo_flags.append('--emit-inference-func-sig')
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

    ret = run_shell_cmd(
        ['g++', *flags, test_file,  onnx_pb_cc_file,
         obj_file, bin_file, *include_path, *lib_path,
         '-o', exe_file, '-Wno-deprecated-declarations'])
    if ret[0]:
        return [None, *ret]
    return [exe_file, *ret]

# single case test


results = []


def print_results():
    print(results)
    passed = list(filter(lambda x: x[2], results))
    failed = list(filter(lambda x: not x[2], results))
    print('Total tests: {0}'.format(len(results)))
    print('Passed: {0}'.format(len(passed)))
    print('Failed: {0}'.format(len(failed)))

    for f in failed:
        print('Failed test {0} [{1}]\n\t{2}\n\t{3}'.format(
            f[0], f[1], f[3][1], f[3][3]))


def single_test(test_case, device, error_thr, enable_timeperf, data_path, sema = None):
    if not sema:
        sema = contextlib.suppress()
    with sema:
        print('----------Test: [' + test_case + ':' + device + ']------')
        case_path = os.path.join(data_path, test_case)
        flags = ['-g']
        if enable_timeperf == 'TRUE':
            flags.append('-DTIME_PERF')
        ret = compile(case_path)
        obj_file = ret[0]
        if not obj_file:
            results.append([test_case, device, False, ret[-1:]])
            return False
        ret = link(case_path, obj_file, device, flags)
        exe_file = ret[0]
        if not exe_file:
            results.append([test_case, device, False, ret[-1:]])
        ret = run_shell_cmd([exe_file, str(error_thr), '0', device, case_path])
        if ret[0]:
            results.append([test_case, device, False, ret])
        else:
            results.append([test_case, device, True, ret])
        return ret
    # all cases test


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

    flags = ['-g']
    if args.enable_time_perf:
        flags.append('-DTIME_PERF')

    threads = []
    for test_case in list_cases:
        print('----------Test: [' + test_case + ':' + str(devices) + ']------')
        error_thr = args.error_threshold
        for device in devices:
            if (args.enable_time_perf):
                single_test(test_case, device, error_thr,
                            args.enable_time_perf, data_path)
            else:
                t = threading.Thread(target=single_test, args=[test_case, device, error_thr,
                                                               args.enable_time_perf, data_path])
                threads.append(t)
                t.start()
    for t in threads:
        t.join(timeout=300)
# test according to the config.csv


def list_test(data_path):
    config_file = os.path.join(test_path, 'config.csv')
    threads = []
    pool_sema = threading.BoundedSemaphore(value = 16)
    with open(config_file) as csvfile:
        reader = csv.reader(csvfile)
        for index, row in enumerate(reader):
            test_name = row[0]
            device = row[1]
            error_thre = row[2]
            timed_test = row[3]
            if index != 0 and row[5] == 'yes' and row[6] == 'PASS':
                parallel = timed_test != 'TRUE'
                if not parallel:
                    # Wait for all existing threads to finish
                    for t in threads:
                        t.join()
                t = threading.Thread(target=single_test, args=['test_' + test_name,
                                                               device, error_thre, timed_test, data_path, pool_sema])
                threads.append(t)
                t.start()
                if not parallel:
                    t.join()
    for t in threads:
        t.join()
    print_results()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-mode', type=str,
                        help='three test modes: single, all, list')
    parser.add_argument('--enable-time-perf', type=bool,
                        default=False, help='whether it takes time test')
    parser.add_argument('--test-case', type=str,
                        help='input test case name, only used in single case test')
    parser.add_argument('--error-threshold', type=float,
                        help='comparison error with expected result')
    parser.add_argument(
        '--device', type=str, help='ODLA supported backend: dnnl, eigen, tensorrt, xnnpack')
    return parser


if __name__ == "__main__":
    args = get_args().parse_args()

    data_path = os.path.join(test_path, 'data')
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
