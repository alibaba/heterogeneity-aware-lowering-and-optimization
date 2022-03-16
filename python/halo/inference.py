# Copyright (C) 2019-2021 Alibaba Group Holding Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http:#www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#! /usr/bin/env python3

from halo import halo
from halo import odla
from pathlib import Path
import sys
import logging
from logging import StreamHandler, Formatter
import os


class Inference:
    def __init__(
        self,
        model_file,
        input_shapes,
        output_names,
        device,
        batch,
        format,
        qps,
        debug,
        log_level,
        model_type=""
    ):
        self.debug = debug
        logging.getLogger("halo").setLevel(log_level)
        self.logger = logging.getLogger(__name__)
        self.model_file = model_file
        self.input_shapes = input_shapes
        self.output_names = output_names
        if not format:
            suffixes = {
                ".onnx": "ONNX",
                ".pb": "TENSORFLOW",
                ".tflite": "TFLITE",
                ".caffemodel": "CAFFE",
            }
            suffixes.setdefault("INVALID")
            suffix = Path(model_file).suffix
            format = suffixes[suffix]
        self.format = format
        self.device = device
        self.batch = batch
        self.qps = qps
        self.model_type = model_type
        self.model = None
        self.so_file = None

    def __del__(self):
        del self.model

    def Initialize(self):
        self.logger.info(f"Begin initialization;{self.model_file}")
        self.so_file = "/usr/local/lib/libvodla.so"
        self.model = odla.ODLAModel(self.so_file)
        self.model.Load(
            self.model_file,
            self.input_shapes,
            self.output_names,
            self.format,
            self.batch,
            self.qps,
            self.model_type)
        self.logger.info("Done initialization")

    def Run(self, data):
        if self.model is None:
            self.Initialize()
        return self.model.Execute(data, self.model_file, self.batch)
