#!/usr/bin/env python3
import torch
import torch.onnx

model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch',
                       'unet', in_channels=3, out_channels=1, init_features=32,
                       pretrained=True)

torch.onnx.export(model, torch.rand(1, 3, 256, 256), 'unet.onnx',
                  export_params=True, opset_version=10, do_constant_folding=True,
                  input_names=['input'],
                  output_names=['output'])
