import torch
import torch.onnx
import sys

if __name__ == "__main__":
    url = 'mateuszbuda/brain-segmentation-pytorch'
    name = 'unet'
    file_path = sys.argv[1]

    model = torch.hub.load(url, name,
                           in_channels=3, out_channels=1, init_features=32, pretrained=True)
    torch.onnx.export(model, torch.rand(1, 3, 256, 256), file_path, export_params=True, opset_version=10, do_constant_folding=True,
                      input_names=['input'], output_names=['output'])
