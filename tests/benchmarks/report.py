#!/usr/bin/env python3

"""
Copyright (C) 2019-2021 Alibaba Group Holding Limited.

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
import contextlib
import subprocess
import os
import numpy as np


def verify(proc):
    if proc.returncode != 0:
        print(proc.args)
        print(proc.stderr)
        exit(proc.returncode)

@contextlib.contextmanager
def keep_curdir():
    curdir = os.getcwd()
    try:
        yield
    finally:
        os.chdir(curdir)

def lit_test(test_dir):
    with keep_curdir():
        if not os.path.isfile(os.path.join(test_dir, 'build.ninja')):
            print('build directory %s is empty' % test_dir)
            return 0
        os.chdir(test_dir)

        cmd = ['ninja', 'clean']
        verify(subprocess.run(cmd))

        cmd = ['ninja', 'check-halo-MZBenchs']
        verify(subprocess.run(cmd))

def gen_mkdown_tb(head, rows, cols, data):
    height, width = data.shape
    lines = ["\n"]

    lines += ["| {} | {} |".format(head, ' | '.join(cols))]

    line = "| {} |".format(":{}".format('-' * len(head)))
    for i in range(width):
        line = "{} {} |".format(line, ":{}".format('-' * len(cols[i])))
    lines += [line]
    
    for i in range(height):
        d = list(map(str, list(data[i])))
        lines += ["| {} | {} |".format(rows[i], ' | '.join(d))]

    table = '\n'.join(lines)
    return table

def write_mkdown(build_dir, mkdown_dir):
    mkdown_file = os.path.join(mkdown_dir, 'model_zoo.md')

    path = os.getcwd()
    dirnames = os.listdir(path)
    devices = []
    models = []
    for dev_name in dirnames:
        if (os.path.isdir(dev_name)):
            subdirnames = os.listdir(dev_name)
            devices.append(dev_name)
            for model_name in subdirnames:
                if (os.path.isdir(os.path.join(dev_name,model_name))):
                    models.append(model_name)
    models = list(set(models))
    
    rows = len(devices)
    cols = len(models) 
    data = np.zeros((rows, cols))
    
    for row_id, device in enumerate(devices):
        for col_id, model in enumerate(models):
            log_file = os.path.join(build_dir, device, model, 'log.txt')
            with open(log_file, 'r') as log:
                lines = log.readlines()
                print(lines[1].split()[2])
                data[row_id][col_id] = float(lines[1].split()[2])
                
    mkdown_tb = "\n#### ODLA Perf Data"
    mkdown_tb += gen_mkdown_tb("Devices\\Models(seconds)", devices, models, data)
    with open(mkdown_file, 'a') as mkdown:
        mkdown.write(mkdown_tb)

halo_root = os.path.join(os.getcwd(), '..', '..')
build_dir = os.path.join(halo_root, 'build', 'tests', 'benchmarks')
mkdown_dir = os.path.join(halo_root, 'models', 'benchmarks')
test_dir = os.path.join(halo_root, 'build')

lit_test(test_dir)
write_mkdown(build_dir, mkdown_dir)