from __future__ import absolute_import

import torch
import numpy as np
import onnx
from google.protobuf.json_format import MessageToDict
import torchvision.models as models

onnx_datatype = onnx.TensorProto.FLOAT16
MODEL_LIST = [
  # efficientnet
  'efficientnet-v1-b0', 'efficientnet-v1-b4', 'efficientnet-v1-b5', 'efficientnet-v1-b6',
  # mobilenet
  'mobilenet-v2', 'mobilenet-v2-1.4', 'mobilenet-v3-small', 'mobilenet-v3-large',
  # resnet
  'resnet-18', 'resnet-34', 'resnet-50',
  # resnext
  'resnext-50',
  # inception
  'inception-v3',
  # shufflenet
  'shufflenet-v2-x0.5', 'shufflenet-v2-x1.0', 'shufflenet-v2-x2.0',
  # mnasnet
  'mnasnet-0.5', 'mnasnet-1.0', 'mnasnet-1.3',
  # vgg
  'vgg-16',
  # regnet
  'regnet_y_400mf', 'regnet_y_800mf', 'regnet_y_1_6gf', 'regnet_y_3_2gf', 'regnet_y_8gf', 'regnet_y_16gf', 'regnet_y_32gf', 'regnet_y_128gf', 'regnet_x_400mf', 'regnet_x_800mf', 'regnet_x_1_6gf', 'regnet_x_3_2gf', 'regnet_x_8gf', 'regnet_x_16gf', 'regnet_x_32gf', 'regnet_x_128gf',
  # vision transformer
  'vit-b-16', 'vit-l-16',
  # convnext
  'convnext-tiny', 'convnext-small', 'convnext-base', 'convnext-large',
  # bert
  'bert-large-1x64', 'bert-large-1x3',
  # test
  'toy', 'memopt',
]

GVN = -1
PAR_EXEC_ID = 0

def gvn(reset=False):
  global GVN

  if reset:
    GVN = -1
    return None

  GVN += 1
  return GVN

def to_numpy(tensor):
  return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def par_exec_id(increment=False):
  global PAR_EXEC_ID
  if increment:
    PAR_EXEC_ID += 1
  return PAR_EXEC_ID

def activation_type(graph, node):
  nodes = find_nodes_by_arg_name(graph, node.output[0])
  nodes.remove(node)

  if len(nodes) == 1:
    if nodes[0].op_type in ["HardSigmoid", "Sigmoid", "Relu", "Clip"]:
      # TODO: tvm bug
      if nodes[0].op_type == "HardSigmoid":
        return "Sigmoid"
      return nodes[0].op_type
  elif len(nodes) == 2:
    if nodes[0].op_type == "Sigmoid" and nodes[1].op_type == "Mul":
      nodes_ = find_nodes_by_arg_name(graph, nodes[0].output[0])
      for n in nodes:
        nodes_.remove(n)
      assert len(nodes_) == 0
      return "SiLU"
    elif nodes[0].op_type == "Mul" and nodes[1].op_type == "Sigmoid":
      nodes_ = find_nodes_by_arg_name(graph, nodes[1].output[0])
      for n in nodes:
        nodes_.remove(n)
      assert len(nodes_) == 0
      return "SiLU"
    elif nodes[0].op_type == "HardSigmoid" and nodes[1].op_type == "Mul":
      nodes_ = find_nodes_by_arg_name(graph, nodes[0].output[0])
      for n in nodes:
        nodes_.remove(n)
      assert len(nodes_) == 0
      # TODO: tvm bug
      # return "HardSiLU"
      return "SiLU"
    elif nodes[0].op_type == "Mul" and nodes[1].op_type == "HardSigmoid":
      nodes_ = find_nodes_by_arg_name(graph, nodes[1].output[0])
      for n in nodes:
        nodes_.remove(n)
      assert len(nodes_) == 0
      # TODO: tvm bug
      # return "HardSiLU"
      return "SiLU"
  return "Identity"

def is_silu(graph, nodes):
  if len(nodes) != 2:
    return False

  n1, n2 = nodes

  n1_inputs = find_nodes_by_arg_name(graph, n1.input[0])
  n1_inputs.remove(n1)
  n2_inputs = find_nodes_by_arg_name(graph, n2.input[0])
  n2_inputs.remove(n2)

  if n1.op_type == "Sigmoid":
    outputs = find_nodes_by_arg_name(graph, n1.output[0])
    outputs.remove(n1)
    outputs.remove(n2)
    if len(outputs) > 0:
      return False
    return n2.op_type == "Mul"
  elif n1.op_type == "Mul":
    outputs = find_nodes_by_arg_name(graph, n2.output[0])
    outputs.remove(n1)
    outputs.remove(n2)
    if len(outputs) > 0:
      return False
    return n2.op_type == "Sigmoid"
  else:
    return False

def is_hardsilu(graph, nodes):
  if len(nodes) != 2:
    return False

  n1, n2 = nodes

  n1_inputs = find_nodes_by_arg_name(graph, n1.input[0])
  n1_inputs.remove(n1)
  n2_inputs = find_nodes_by_arg_name(graph, n2.input[0])
  n2_inputs.remove(n2)

  if n1.op_type == "HardSigmoid":
    outputs = find_nodes_by_arg_name(graph, n1.output[0])
    outputs.remove(n1)
    outputs.remove(n2)
    if len(outputs) > 0:
      return False
    return n2.op_type == "Mul"
  elif n1.op_type == "Mul":
    outputs = find_nodes_by_arg_name(graph, n2.output[0])
    outputs.remove(n1)
    outputs.remove(n2)
    if len(outputs) > 0:
      return False
    return n2.op_type == "HardSigmoid"
  else:
    return False

def is_silu_like(graph, nodes):
  return is_silu(graph, nodes) or is_hardsilu(graph, nodes)

def silu_like_type(graph, nodes):
  if is_silu(graph, nodes):
    return "Sigmoid"
  elif is_hardsilu(graph, nodes):
    return "HardSigmoid"
  else:
    raise Exception("Not supported type")

def create_initializer_tensor(
        name: str,
        tensor_array: np.ndarray,
        data_type: onnx.TensorProto = onnx.TensorProto.FLOAT16
) -> onnx.TensorProto:
    # (TensorProto)
    initializer_tensor = onnx.helper.make_tensor(
        name=name,
        data_type=data_type,
        dims=tensor_array.shape,
        vals=tensor_array.flatten().tolist())

    return initializer_tensor

def get_arg_shape(graph, node, arg_name):
  dims = None
  for val in graph.value_info:
    val = MessageToDict(val)
    # print(val)
    if val['name'] == arg_name:
      dims = val['type']['tensorType']['shape']['dim']
      dims = [int(e['dimValue']) for e in dims]
      return dims

  if graph.node[0] == node:
    assert len(graph.input) < 2 # single input
    m_dict = MessageToDict(graph.input[0])
    dim_info = m_dict["type"]["tensorType"]["shape"]["dim"]  # ugly but we have to live with this when using dict
    dims = [int(d.get("dimValue")) for d in dim_info]
    return dims

# NOTE: check result nodes for uses
def find_nodes_by_arg_name(graph, arg_name):
  r = []
  for node in graph.node:
    if arg_name in node.input or arg_name in node.output:
      r.append(node)
  return r

def find_initializer_by_arg_name(graph, arg_name):
  for initializer in graph.initializer:
    if arg_name == initializer.name:
      return initializer

def find_value_info_by_arg_name(graph, arg_name):
  for value_info in graph.value_info:
    if arg_name == value_info.name:
      return value_info

def find_node_index_by_name(graph, name):
  for i, node in enumerate(graph.node):
    if node.name == name:
      return i
  return None

def find_attribute_by_name(node, name):
  for attr in node.attribute:
    if attr.name == name:
      return attr

# assign node the unique name
def preprocess(onnx_model):
  gvn(reset=True)
  for node in onnx_model.graph.node:
    # if not node.name:
    op_type = node.op_type
    if not op_type:
      op_type = "Node"
    node.name = f"{op_type}_{gvn()}"

def supportedActivationNodes(graph, nodes):
  if len(nodes) == 1:
    if nodes[0].op_type in ["Clip", "Relu", "Sigmoid", "HardSigmoid"]:
      return True
  elif len(nodes) == 2:
    return is_silu_like(graph, nodes)
  return False

class Net(torch.nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.conv1 = torch.nn.Conv2d(3, 64, 1, padding=0, stride=1, bias=True)
    self.act1 = torch.nn.Hardtanh(0, 6)
    self.conv2 = torch.nn.Conv2d(64, 64, 3, padding=1, stride=1, groups=64, bias=True)
    self.act2 = torch.nn.Hardtanh(0, 6)
    self.conv3 = torch.nn.Conv2d(64, 64, 1, padding=0, stride=1, bias=True)
    self.act3 = torch.nn.Hardtanh(0, 6)
    self.gap = torch.nn.AvgPool2d(16)
    self.flatten = torch.nn.Flatten()
    self.linear = torch.nn.Linear(64, 10)

  def forward(self, x):
    x = self.conv1(x)
    x = self.act1(x)
    # x_ = torch.sigmoid(x)
    # x = torch.mul(x, x_)
    x = self.conv2(x)
    x = self.act2(x)
    # x_ = torch.sigmoid(x)
    # x = torch.mul(x, x_)
    x = self.conv3(x)
    x = self.act3(x)
    # x_ = torch.sigmoid(x)
    # x = torch.mul(x, x_)
    x = self.gap(x)
    x = self.flatten(x)
    x = self.linear(x)
    return x

class NetMemOptTest(torch.nn.Module):
  def __init__(self):
    super(NetMemOptTest, self).__init__()
    self.conv1 = torch.nn.Conv2d(3, 64, 3, padding=1, stride=1, bias=True)
    self.act1 = torch.nn.Hardtanh(0, 6)
    self.conv2 = torch.nn.Conv2d(64, 64, 3, padding=1, stride=1, groups=1, bias=True)
    self.act2 = torch.nn.Hardtanh(0, 6)
    self.conv3 = torch.nn.Conv2d(64, 64, 3, padding=1, stride=1, bias=True)
    self.act3 = torch.nn.Hardtanh(0, 6)
    self.gap = torch.nn.AvgPool2d(16)
    self.flatten = torch.nn.Flatten()
    self.linear = torch.nn.Linear(64, 10)

  def forward(self, x):
    x = self.conv1(x)
    x = self.act1(x)
    # x_ = torch.sigmoid(x)
    # x = torch.mul(x, x_)
    x = self.conv2(x)
    x = self.act2(x)
    # x_ = torch.sigmoid(x)
    # x = torch.mul(x, x_)
    x = self.conv3(x)
    x = self.act3(x)
    # x_ = torch.sigmoid(x)
    # x = torch.mul(x, x_)
    x = self.gap(x)
    x = self.flatten(x)
    x = self.linear(x)
    return x

def calc_pads(node):
  dilations=list(find_attribute_by_name(node, 'dilations').ints)
  kernel_shape=list(find_attribute_by_name(node, 'kernel_shape').ints)
  pads=list(find_attribute_by_name(node, 'pads').ints)
  strides=list(find_attribute_by_name(node, 'strides').ints)

  # TODO: currently only support dilation == 1
  for dilation in dilations:
    assert dilation == 1

  assert len(strides) == 2 and strides[0] == strides[1]
  assert pads[0] == pads[2] and pads[0] <= kernel_shape[0] // 2

  pad_start = (pads[0], pads[1], 0, pads[3])
  pad_middle = (0, pads[1], 0, pads[3])
  pad_end = (0, pads[1], pads[2], pads[3])

  return pad_start, pad_middle, pad_end

def calc_space(node, h):
  dilations=list(find_attribute_by_name(node, 'dilations').ints)
  kernel_shape=list(find_attribute_by_name(node, 'kernel_shape').ints)
  pads=list(find_attribute_by_name(node, 'pads').ints)
  strides=list(find_attribute_by_name(node, 'strides').ints)

  # TODO: currently only support dilation == 1
  for dilation in dilations:
    assert dilation == 1

  # TODO: currently only support stride <= 2
  for stride in strides:
    assert stride <= 2

  assert len(strides) == 2 and strides[0] == strides[1]
  assert pads[0] == pads[2] and pads[0] <= kernel_shape[0] // 2

  stride = strides[0]

  space = (kernel_shape[0] // 2) * 2 - pads[0]
  if stride == 1:
    space_start = (0, space)
    space_middle = (-space, space)
    space_end = (-space, 0)
  elif stride == 2:
    if h % 2 == 0:
      space_start = (0, 0)
      space_middle = (-space, 0)
      space_end = (-space, 0)
    else:
      space_start = (0, space)
      space_middle = (-space, space)
      space_end = (-space, 0)
  return space_start, space_middle, space_end

def get_torch_model(name):
  model = None
  if name == "efficientnet-v1-b0":
    model = models.efficientnet_b0(pretrained=True)
  elif name == "efficientnet-v1-b4":
    model = models.efficientnet_b4(pretrained=True)
  elif name == "efficientnet-v1-b6":
    model = models.efficientnet_b6(pretrained=True)
  elif name == "mobilenet-v2":
    model = models.mobilenet_v2(pretrained=True)
  elif name == "mobilenet-v2-1.4":
    model = models.mobilenet_v2(pretrained=True, width_mult=1.4)
  elif name == "mobilenet-v3-small":
    model = models.mobilenet_v3_small(pretrained=True)
  elif name == "mobilenet-v3-large":
    model = models.mobilenet_v3_large(pretrained=True)
  elif name == "resnet-18":
    model = models.resnet18(pretrained=True)
  elif name == "resnet-34":
    model = models.resnet34(pretrained=True)
  elif name == "resnet-50":
    model = models.resnet50(pretrained=True)
  elif name == "resnext-50":
    model = models.resnext50_32x4d(pretrained=True)
  elif name == "inception-v3":
    model = models.inception_v3(pretrained=True)
  elif name == "shufflenet-v2-x0.5":
    model = models.shufflenet_v2_x0_5(pretrained=True)
  elif name == "shufflenet-v2-x1.0":
    model = models.shufflenet_v2_x1_0(pretrained=True)
  elif name == "shufflenet-v2-x2.0":
    model = models.shufflenet_v2_x2_0(pretrained=False) # pretrained model is not yet supported
  elif name == "mnasnet-0.5":
    model = models.mnasnet0_5(pretrained=True)
  elif name == "mnasnet-1.0":
    model = models.mnasnet1_0(pretrained=True)
  elif name == "mnasnet-1.3":
    model = models.mnasnet1_3(pretrained=False) # pretrained model is not yet supported
  elif name == "vgg-16":
    model = models.vgg16(pretrained=True)
  elif name == "regnet_y_400mf":
    model = models.regnet_y_400mf(pretrained=True)
  elif name == "regnet_y_800mf":
    model = models.regnet_y_800mf(pretrained=True)
  elif name == "regnet_y_1_6gf":
    model = models.regnet_y_1_6gf(pretrained=True)
  elif name == "regnet_y_3_2gf":
    model = models.regnet_y_3_2gf(pretrained=True)
  elif name == "regnet_y_8gf":
    model = models.regnet_y_8gf(pretrained=True)
  elif name == "regnet_y_16gf":
    model = models.regnet_y_16gf(pretrained=True)
  elif name == "regnet_y_32gf":
    model = models.regnet_y_32gf(pretrained=True)
  elif name == "regnet_y_128gf":
    model = models.regnet_y_128gf(pretrained=True)
  elif name == "regnet_x_400mf":
    model = models.regnet_x_400mf(pretrained=True)
  elif name == "regnet_y_800mf":
    model = models.regnet_x_800mf(pretrained=True)
  elif name == "regnet_x_1_6gf":
    model = models.regnet_x_1_6gf(pretrained=True)
  elif name == "regnet_x_3_2gf":
    model = models.regnet_x_3_2gf(pretrained=True)
  elif name == "regnet_x_8gf":
    model = models.regnet_x_8gf(pretrained=True)
  elif name == "regnet_x_16gf":
    model = models.regnet_x_16gf(pretrained=True)
  elif name == "regnet_x_32gf":
    model = models.regnet_x_32gf(pretrained=True)
  elif name == "regnet_x_128gf":
    model = models.regnet_x_128gf(pretrained=True)
  elif name == "vit-b-16":
    model = models.vit_b_16(pretrained=True)
  elif name == "vit-l-16":
    model = models.vit_l_16(pretrained=True)
  elif name == "convnext-tiny":
    model = models.convnext_tiny(pretrained=True)
  elif name == "convnext-small":
    model = models.convnext_small(pretrained=True)
  elif name == "convnext-base":
    model = models.convnext_base(pretrained=True)
  elif name == "convnext-large":
    model = models.convnext_large(pretrained=True)
  elif name == "toy":
    model = Net()
  elif name == "memopt":
    model = NetMemOptTest()
  else:
    raise Exception(f"Unsupported model: {name}")
  return model

def get_random_input(name):
  x = torch.randn(1, 3, 224, 224).cuda()
  if name == "efficientnet-v1-b6":
    x = torch.randn(1, 3, 528, 528).cuda()
  elif name == "inception-v3":
    x = torch.randn(1, 3, 299, 299).cuda()
  elif name in ["toy", "memopt"]:
    x = torch.randn(1, 3, 16, 16).cuda()
  return x

def parse_kernel_number(l):
  return int(l.split("-")[1].split(".")[0])

def get_kernel_start_and_end(trace_path, n_run=3):
  start=None

  # start, end
  runs = []
  skip = True # skip first interval
  with open(f"./{trace_path}/stats.csv") as f:
    lines = f.readlines()

    for i, l in enumerate(lines):
      if i == 0:
        continue

      l = l.split(",")

      n = parse_kernel_number(l[0].strip())
      name = l[1].strip()

      if name.find("forward_kernel_cuda_start") != -1:
        # if skip:
        #   skip = False
        #   continue

        start = n

      if start is not None and name.find("forward_kernel_cuda_end") != -1:
        runs.append((start, n))

  intv = runs[0][1] - runs[0][0]
  for s, t in runs:
    assert intv == t - s

  assert len(runs) == n_run

  return (runs[2][0]+1, runs[2][1]-1)
