import torch
import torch.onnx
import sys
import onnx

def change_input_dim(model, batch_size):
    # The following code changes the first dimension of every input to be batch_size
    # Modify as appropriate ... note that this requires all inputs to
    # have the same batch_size
    inputs = model.graph.input
    # print(inputs[0])
    # print(inputs[0].type.tensor_type.shape.dim[0])
    for input in inputs:
        # Checks omitted.This assumes that all inputs are tensors and have a shape with first dim.
        # Add checks as needed.
        dim1 = input.type.tensor_type.shape.dim[0]
        # print(dim1)
        # update dim to be a symbolic value
        if isinstance(batch_size, str):
            # set dynamic batch size
            dim1.dim_param = batch_size
        elif (isinstance(batch_size, str) and batch_size.isdigit()) or isinstance(batch_size, int):
            # set given batch size
            dim1.dim_value = int(batch_size)
        else:
            # set batch size of 1
            dim1.dim_value = 1


def apply(transform, infile, outfile, batch_size):
    model = onnx.load(infile)
    transform(model, batch_size)
    onnx.save(model, outfile)
if __name__ == "__main__":
    name = sys.argv[1]
    inputfile = name + '.onnx'
    for i in range(1, 7):
        bs = 2**i
        outputfile = name + '_bs' + str(bs) + '.onnx'
        apply(change_input_dim, inputfile, outputfile, bs)
# rebatch('resnet50-v2-7.onnx', 'resnet50-v2-7_bs4.onnx', '4')
    # def read_onnx(model):
    #     model = onnx.load(model)
    #     graph = model.graph
    #     node = graph.node
    #     print(node[2469:2569])
    # read_onnx(inputfile)