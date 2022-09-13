#!/usr/bin/env python3
import os
import os.path
from functools import reduce
import ctypes
import numpy as np
from PIL import Image


base_path = os.path.dirname(os.path.abspath(__file__))
output_path = os.getenv("RETINANET_OUTPUT_PATH")
model_path = os.getenv("RETINANET_MODEL_PATH")

image_path = os.path.join(model_path, "demo.jpg")
ref_output_path = os.path.join(base_path, "ref_out")

so_exe = ctypes.CDLL(os.path.join(output_path, "retinanet-9.so"))


def flatten(inputs):
    return [[flatten(i) for i in inputs] if isinstance(inputs, (list, tuple)) else inputs]


def update_flatten_list(inputs, res_list):
    for i in inputs:
        res_list.append(i) if not isinstance(i, (list, tuple)) else update_flatten_list(i, res_list)
    return res_list


def to_numpy(x):
    if type(x) is not np.ndarray:
        x = x.detach().cpu().numpy() if x.requires_grad else x.cpu().numpy()
    return x
    

def run_inference(image_arr, is_save=False):
    image_arr = image_arr.flatten().astype(ctypes.c_float)
    
    if is_save:
        np.savetxt(os.path.join(output_path, "input_0.txt"), 
                image_arr.flatten().astype(np.float32), 
                fmt="%.17f", 
                delimiter=',', 
                encoding='utf-8')

    def struct_out(out):
        return (ctypes.c_float * reduce(lambda x, y: x * y, out))()

    outputs_shape = [(1, 720, 60, 80), (1, 720, 30, 40), (1, 720, 15, 20), (1, 720, 8, 10), (1, 720, 4, 5), 
                    (1, 36, 60, 80), (1, 36, 30, 40), (1, 36, 15, 20), (1, 36, 8, 10), (1, 36, 4, 5)]
    outputs = [struct_out(o) for o in outputs_shape]

    # run inference 
    so_exe.model(image_arr.ctypes.data_as(ctypes.c_void_p), outputs[0], outputs[1], outputs[2], 
        outputs[3], outputs[4], outputs[5], outputs[6], outputs[7], outputs[8], outputs[9])

    for out in outputs:
        _data = np.ctypeslib.as_array(out).astype(np.float32)
        _index = outputs.index(out)
        _shape = outputs_shape[_index]
        outputs[_index] = np.reshape(_data, _shape)

        output_filename = f"output_{_shape[1]}_{_shape[-1]}.txt"
        if is_save:
            np.savetxt(os.path.join(output_path, output_filename), 
                _data.flatten(), 
                fmt="%.17f", 
                delimiter=',', 
                encoding='utf-8')

        # Checking model output
        ref_output_file = os.path.join(ref_output_path, f"{output_filename}")
        diff_result = np.allclose(_data.flatten(), np.loadtxt(ref_output_file, dtype=np.float32), rtol=1e-4, atol=1e-4)
        print(f"{output_filename} {diff_result}")

    return outputs


def preprocess(is_preprocessed=True):
    _preprocess_data = None
    if is_preprocessed:
        # read input data preprocessed from txt file
        _preprocess_data = np.loadtxt(os.path.join(ref_output_path, "input_0.txt"), dtype=np.float32)
    else:
        # preprocess image
        from torchvision import transforms

        input_image = Image.open(image_path)
        prepro = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = prepro(input_image)
        _preprocess_data = input_tensor.unsqueeze(0).numpy()

    return _preprocess_data


if __name__ == "__main__":

    input_data = preprocess()
    run_inference(input_data, is_save=True)