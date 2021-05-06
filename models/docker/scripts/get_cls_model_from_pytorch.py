#! /usr/bin/env python3
import torch
import torch.onnx
import sys

if __name__ == "__main__":
    name = sys.argv[1]
    file_path = name + '.onnx'
    if len(sys.argv) > 2:
        file_path = sys.argv[2]
    h = int(sys.argv[3]) if len(sys.argv) > 3 else 224
    w = int(sys.argv[4]) if len(sys.argv) > 4 else h
    url = 'pytorch/vision:v0.6.0'
    model = torch.hub.load(url, name, pretrained=True)
    torch.onnx.export(model, torch.rand(1, 3, h, w), file_path, export_params=True,
                      opset_version=10, do_constant_folding=True,
                      input_names=['input'], output_names=['output'])
