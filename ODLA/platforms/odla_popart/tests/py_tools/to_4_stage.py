import onnx
import onnx.helper as helper
import re


name_lists = [
    [],
    [
    "reload_query/bert/encoder/layer_[1-9]/.",
    "reload_query/bert/encoder/layer_1[01]/.",
    "reload_query/bert/encoder/Reshape_13",
    "reload_query/bert/pooler",
    "reload/NotEqual",
    "reload/Cast",
    "reload/Sum$",
    "reload/Sum/reduction_indices",
    "reload/bert/encoder/mul",
    "reload/bert/encoder/Shape_2",
    "reload/bert/encoder/strided_slice_2",
    "reload/bert/encoder/Shape",
    "reload/SequenceMask",
    "reload/bert/encoder/strided_slice",
    "reload/bert/encoder/Reshape$",
    "reload/bert/encoder/Reshape_1$",
    "reload/bert/encoder/Reshape_1/shape$",
    "reload/bert/encoder/ones",
    "reload/bert/encoder/Reshape/shape",
    "reload/bert/embeddings",
    "reload/bert/encoder/layer_0/",
    "reload/bert/encoder/layer_1/",
    ],
    [
    "reload/bert/encoder/layer_[2-6]/."
    ],
    [
    "reload/bert/encoder/layer_[7-9]/.",
    "reload/bert/encoder/layer_1[01]/.",
    "reload/bert/encoder/Reshape_13",
    "reload/bert/pooler",
    "reload/semantics_cnn",
    "query_attenqt_cls/.",
    "reload/Sum_1",
    "sim_qt_cls/.",
    "query_attenqt_token/.",
    "sim_qt_token/.",
    "query_attenqc_cls/.",
    "sim_qc_cls/.",
    "query_attenqc_token/.",
    "sim_qc_token/.",
    "^merge_sim/.",
    "^qtc_sim/.",
    "^qc_sim/."
    "^Softmax",
    "^mul/y",
    "^mul",
    "^Sum/reduction_indices",
    "^Sum",
    "^Softmax_1",
    "^mul_1/y",
    "^mul_1",
    "^Sum_1/reduction_indices",
    "^Sum_1",
    "^Softmax_2",
    "^mul_2/y",
    "^mul_2",
    "^Sum_2/reduction_indices",
    "^Sum_2",
    "^Reshape/shape",
    "^Reshape",
    "^Reshape_1/shape",
    "^Reshape_1",
    "^Reshape_2/shape",
    "^Reshape_2",
    "^concat/axis",
    "^concat_pre_end",
    "^concat_wrapped",
    "^NoOp",
    "^Identity",
    "^output0",
    "^output_0",
    "^concat"
    ]
]    

model = onnx.load('pipelined.onnx')
graph = model.graph
nodes = graph.node
for node in nodes:
    matched = False
    num = -1
    for i in range(4):
        for pattern in name_lists[i]:
            pattern = pattern.replace('/', '_')
            #print(f'The pattern is: {pattern}')
            if re.search(pattern, node.name):
                matched = True
                num = i
                break
        if matched:
            break
    if not matched:
        num = 0
    attributes = node.attribute
    for attribute in attributes:
        if attribute.name == "__pipeline_stage":
            print(f'{node.name} set pipeline stage as {num}')
            attribute.i = num
        if attribute.name == "__ipu_number":
            print(f'{node.name} will be placed on IPU {num % 2}')
            attribute.i = num % 2


onnx.checker.check_model(model)
onnx.save(model, 'pipeline_4_stage.onnx')
