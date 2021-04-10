from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import models
import dataset
from config import update_config
from config import cfg
import subprocess
import argparse
import os
import pprint
import sys
import gdown

import torch
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

import numpy as np


def dump_iodata(file_name, datas, array_name):
    with open(file_name, 'w') as cf:
        array_name = 'float ' + array_name + ' [' + str(len(datas)) + '] = {\n'
        cf.write(array_name)
        for data in datas:
            # print(data.item())
            cf.write(str(data.item()) + ',\n')
        cf.write('};\n')


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    update_config(cfg, args)

    model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=False
    )

    model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE,
                                     map_location=torch.device('cpu')), strict=False)
    model.eval()

    # pytorch to onnx
    h = 256
    w = 256
    export_onnx_file = cfg.TEST.MODEL_FILE.split('/')[-1].split('.')[0]+".onnx"
    torch.onnx.export(model, torch.rand(1, 3, h, w), export_onnx_file, export_params=True,
                      opset_version=10, do_constant_folding=True,
                      input_names=['input'], output_names=['output'])

    # Data loading code
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    valid_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=False
    )
    with torch.no_grad():
        for i, (input, target, target_weight, meta) in enumerate(valid_loader):
            if i == 0:
                outputs = model(input)
                #print("input tensor")
                dump_iodata("input.in", input[...].flatten(), 'data')
                #print("output tensor")
                if isinstance(outputs, list):
                    output = outputs[-1]
                else:
                    output = outputs
                dump_iodata("output.in", output[...].flatten(), 'output_gold')


if __name__ == '__main__':
    main()
