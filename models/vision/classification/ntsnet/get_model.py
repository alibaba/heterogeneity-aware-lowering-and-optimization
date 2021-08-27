import torch 
import torch.onnx
model = torch.hub.load('nicolalandro/ntsnet-cub200', 'ntsnet', pretrained=True)
torch.onnx.export(model, torch.rand(1, 3, 448, 448), 'ntsnet.onnx', export_params=True,
                    opset_version=10, do_constant_folding=True,
                    input_names=['input'], output_names=['output'])

# RuntimeError: ONNX export failed on an operator with unrecognized namespace torchvision::nms. If you are trying to export a custom operator, make sure you registered it with the right domain and version.