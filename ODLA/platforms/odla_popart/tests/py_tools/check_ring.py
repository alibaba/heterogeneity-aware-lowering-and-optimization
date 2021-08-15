import onnx

model = onnx.load("pipeline_4_stage.onnx")
graph = model.graph
nodes = graph.node


node_lists = [[], [], [], []]
input_lists = [[], [], [], []]
output_lists = [[], [], [], []]

for node in nodes:
  attrs = node.attribute
  for attr in attrs:
    if attr.name == '__pipeline_stage':
      stage = attr.i
    if attr.name == '__ipu_number':
      num = attr.i
  assert num == stage % 2, f'{node.name}: num:{num} can not match stage:{stage}'
  node_lists[stage].append(node)
  for input_ in node.input:
    input_lists[stage].append(input_)
  for output_ in node.output:
    output_lists[stage].append(output_)

for i in range(4):
  print('='*120)
  set_in = set(input_lists[i])
  set_out = set(output_lists[i])
  r_in = set_in - set_out
  print(f'The real input of stage {i} is: {r_in}')
  previous_out = []
  found = False
  for j in range(i):
    previous_out.extend(output_lists[j])
  previous_out = set(previous_out)
  should_empty = r_in - previous_out
  if should_empty:
    print(f'Stage {i} found possible ring: {should_empty}')
  for name in should_empty:
    if name.startswith('tensordict'):
      continue
    for idx in range(4):
      if name in output_lists[idx]:
        print(f'@@@ needed input:{name} found in stage:{idx}')
        break
    for node in node_lists[idx]:
      for output_ in node.output:
        if output_ == name:
          print(f'@@@@@ {name} is output of node: {node.name} of stage:{idx}')
          break
    for node in node_lists[i]:
      for input_ in node.input:
        if input_ == name:
          stage = -1
          num = -1
          for attr in node.attribute:
            if attr.name == '__pipeline_stage':
              stage = attr.i
            if attr.name == '__ipu_number':
              num = attr.i
          print(f'$$$ {name} is input of node: {node.name} on ipu:{num} stage:{stage}')
          # don't break, need all
