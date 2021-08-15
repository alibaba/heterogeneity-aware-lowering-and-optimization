import onnx
import onnx.helper as helper
import re

model = onnx.load('pipeline_test.onnx')
graph = model.graph
nodes = graph.node
for node in nodes:
    has_ipu_number = False
    has_pipeline_stage = False
    attributes = node.attribute
    for attribute in attributes:
        if attribute.name == "__pipeline_stage":
            has_pipeline_stage = True
        if attribute.name == "__ipu_number":
            has_ipu_number = True
        if has_pipeline_stage and has_ipu_number:
            break

    if re.search('^reload_qt', node.name) or re.search('^reload_query', node.name):
        ipu_number = 0
        print(f'{node.name} will be placed to ipu 0')
    else:
        print(f'{node.name} will be placed to ipu 1')
        ipu_number = 1
    if not has_ipu_number:
        print(f'[{node.name}] has not ipu number set')
        new_attr = helper.make_attribute("__ipu_number", ipu_number)
        node.attribute.append(new_attr)
    if not has_pipeline_stage:
        print(f'[{node.name}] has not pipeline stage')
        new_attr = helper.make_attribute("__pipeline_stage", ipu_number)
        node.attribute.append(new_attr)

onnx.checker.check_model(model)
onnx.save(model, 'pipelined.onnx')
