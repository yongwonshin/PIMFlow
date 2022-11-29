import torch
import numpy as np
import torchvision.models as models
import onnx
from google.protobuf.json_format import MessageToDict
from pim.util import Net, activation_type, MODEL_LIST
import os

import argparse

class Range(object):
  def __init__(self, start, end):
    self.start = start
    self.end = end
  def __eq__(self, other):
    return self.start <= other <= self.end

parser = argparse.ArgumentParser()
parser.add_argument("--model", help="model", choices=MODEL_LIST, required=True)
parser.add_argument("--n_channel", type=int, default=16)
parser.add_argument("--split_ratio", type=int,  required=True)
parser.add_argument("--full", action="store_true")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"]=f"{args.n_channel // 4 % 8}"

model = None
if args.model == "efficientnet-v1-b0":
  model = models.efficientnet_b0(pretrained=True)
elif args.model == "efficientnet-v1-b1":
  model = models.efficientnet_b1(pretrained=True)
elif args.model == "efficientnet-v1-b2":
  model = models.efficientnet_b2(pretrained=True)
elif args.model == "efficientnet-v1-b3":
  model = models.efficientnet_b3(pretrained=True)
elif args.model == "efficientnet-v1-b4":
  model = models.efficientnet_b4(pretrained=True)
elif args.model == "efficientnet-v1-b5":
  model = models.efficientnet_b5(pretrained=True)
elif args.model == "efficientnet-v1-b6":
  model = models.efficientnet_b6(pretrained=True)
elif args.model == "efficientnet-v1-b7":
  model = models.efficientnet_b7(pretrained=True)
elif args.model == "mobilenet-v2":
  model = models.mobilenet_v2(pretrained=True)
elif args.model == "mobilenet-v2-1.4": # https://arxiv.org/pdf/1801.04381.pdf
  model = models.mobilenet_v2(pretrained=False, width_mult=1.4)
elif args.model == "mobilenet-v3-small":
  model = models.mobilenet_v3_small(pretrained=True)
elif args.model == "mobilenet-v3-large":
  model = models.mobilenet_v3_large(pretrained=True)
elif args.model == "resnet-18":
  model = models.resnet18(pretrained=True)
elif args.model == "resnet-34":
  model = models.resnet34(pretrained=True)
elif args.model == "resnet-50":
  model = models.resnet50(pretrained=True)
elif args.model == "resnext-50":
  model = models.resnext50_32x4d(pretrained=True)
elif args.model == "inception-v3":
  model = models.inception_v3(pretrained=True)
elif args.model == "shufflenet-v2-x0.5":
  model = models.shufflenet_v2_x0_5(pretrained=True)
elif args.model == "shufflenet-v2-x1.0":
  model = models.shufflenet_v2_x1_0(pretrained=True)
elif args.model == "shufflenet-v2-x2.0":
  model = models.shufflenet_v2_x2_0(pretrained=False) # pretrained model is not yet supported
elif args.model == "mnasnet-0.5":
  model = models.mnasnet0_5(pretrained=True)
elif args.model == "mnasnet-1.0":
  model = models.mnasnet1_0(pretrained=True)
elif args.model == "mnasnet-1.3":
  model = models.mnasnet1_3(pretrained=False) # pretrained model is not yet supported
elif args.model == "vgg-16":
  model = models.vgg16(pretrained=True)
elif args.model == "regnet_y_400mf":
  model = models.regnet_y_400mf(pretrained=True)
elif args.model == "regnet_y_800mf":
  model = models.regnet_y_800mf(pretrained=True)
elif args.model == "regnet_y_1_6gf":
  model = models.regnet_y_1_6gf(pretrained=True)
elif args.model == "regnet_y_3_2gf":
  model = models.regnet_y_3_2gf(pretrained=True)
elif args.model == "regnet_y_8gf":
  model = models.regnet_y_8gf(pretrained=True)
elif args.model == "regnet_y_16gf":
  model = models.regnet_y_16gf(pretrained=True)
elif args.model == "regnet_y_32gf":
  model = models.regnet_y_32gf(pretrained=True)
elif args.model == "regnet_y_128gf":
  model = models.regnet_y_128gf(pretrained=True)
elif args.model == "regnet_x_400mf":
  model = models.regnet_x_400mf(pretrained=True)
elif args.model == "regnet_y_800mf":
  model = models.regnet_x_800mf(pretrained=True)
elif args.model == "regnet_x_1_6gf":
  model = models.regnet_x_1_6gf(pretrained=True)
elif args.model == "regnet_x_3_2gf":
  model = models.regnet_x_3_2gf(pretrained=True)
elif args.model == "regnet_x_8gf":
  model = models.regnet_x_8gf(pretrained=True)
elif args.model == "regnet_x_16gf":
  model = models.regnet_x_16gf(pretrained=True)
elif args.model == "regnet_x_32gf":
  model = models.regnet_x_32gf(pretrained=True)
elif args.model == "regnet_x_128gf":
  model = models.regnet_x_128gf(pretrained=True)
elif args.model == "vit-b-16":
  model = models.vit_b_16(pretrained=True)
elif args.model == "vit-l-16":
  model = models.vit_l_16(pretrained=True)
elif args.model == "swin-b":
  model = models.swin_b()
elif args.model == "swin-s":
  model = models.swin_s()
elif args.model == "convnext-tiny":
  model = models.convnext_tiny(pretrained=True)
elif args.model == "convnext-small":
  model = models.convnext_small(pretrained=True)
elif args.model == "convnext-base":
  model = models.convnext_base(pretrained=True)
elif args.model == "convnext-large":
  model = models.convnext_large(pretrained=True)
elif args.model == "toy":
  model = Net()
elif args.model in ['bert-large-1x1', 'bert-large-1x3', 'bert-large-1x32', 'bert-large-1x64', 'bert-large-1x128', 'bert-base-1x1', 'bert-base-1x3', 'bert-base-1x32', 'bert-base-1x64', 'bert-base-1x128']:
  model = Net() # temporary dummy model
else:
  raise Exception("MUST not reach here!")
model.cuda()
model.eval()
model.half()

# https://smecsm.tistory.com/240
x = torch.randn(1, 3, 224, 224).cuda().half()
# TODO: find all input size for all models
if args.model == "efficientnet-v1-b7":
  x = torch.randn(1, 3, 600, 600).cuda().half()
elif args.model in ["efficientnet-v1-b6", "convnext-large"]:
  x = torch.randn(1, 3, 528, 528).cuda().half()
elif args.model == "efficientnet-v1-b5":
  x = torch.randn(1, 3, 456, 456).cuda().half()
elif args.model == "efficientnet-v1-b4":
  x = torch.randn(1, 3, 380, 380).cuda().half()
elif args.model == "efficientnet-v1-b3":
  x = torch.randn(1, 3, 300, 300).cuda().half()
elif args.model == "efficientnet-v1-b2":
  x = torch.randn(1, 3, 260, 260).cuda().half()
elif args.model == "efficientnet-v1-b1":
  x = torch.randn(1, 3, 240, 240).cuda().half()
elif args.model == "inception-v3":
  x = torch.randn(1, 3, 299, 299).cuda().half()


if args.model in ['bert-large-1x1', 'bert-large-1x3', 'bert-large-1x32', 'bert-large-1x64', 'bert-large-1x128', 'bert-base-1x1', 'bert-base-1x3', 'bert-base-1x32', 'bert-base-1x64', 'bert-base-1x128']:
  onnx_model = onnx.load(f"{args.model}_{args.n_channel}.onnx")
else:
  torch_out = model(x)
  # Export the model
  torch.onnx.export(model,               # model being run
                    x,                         # model input (or a tuple for multiple inputs)
                    f"{args.model}_{args.n_channel}.onnx",   # where to save the model (can be a file or file-like object)
                    export_params=True,        # store the trained parameter weights inside the model file
                    opset_version=13,          # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    # training=TrainingMode.TRAINING,
                    input_names = ['input'],   # the model's input names
                    output_names = ['output']) # the model's output names
                    # dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                    #               'output' : {0 : 'batch_size'}})

  onnx_model = onnx.load(f"{args.model}_{args.n_channel}.onnx")
onnx.checker.check_model(onnx_model)

from pim.util import find_initializer_by_arg_name, get_arg_shape

class OperatorPrinter:
  def __init__(self, model_name):
    self.conv_path = f"{model_name}_conv.csv"
    self.matmul_path = f"{model_name}_matmul.csv"
    with open(self.conv_path, 'w') as f:
      pass
    #   f.write("kernel_name,N,I_c,H,W,O_c,kernel_size,pads,strides,group,dilations,bias,activation\n")
    with open(self.matmul_path, 'w') as f:
      pass
    #   f.write("kernel_name,row,col,bias,activation\n")
    self.conv_configs = set()
    self.fc_configs = set()
    self.pim_configs = set()

  def reset(self):
    self.conv_configs.clear()

  def print(self, op_type, config, debug=False):
    if op_type == "Conv":
      c = f"{config['input_shape'][0]},{config['input_shape'][1]},{config['input_shape'][2]},{config['input_shape'][3]},{config['weight_shape'][0]},\"({config['kernel_shape'][0]},{config['kernel_shape'][1]})\",\"({config['pads'][0]},{config['pads'][1]})\",{config['strides'][0]},{config['group']},{config['dilations'][0]},{int(config['bias'])},{config['activation']}"
      if not args.full:
        if config['kernel_name'].endswith("_pim_added"):
          if c in self.pim_configs:
            return
          else:
            self.pim_configs.add(c)
        else:
          if c in self.conv_configs:
            return
          else:
            self.conv_configs.add(c)

      with open(self.conv_path, 'a') as f:
        f.write(f"{config['kernel_name']},{c}\n")
        if debug:
          print(f"Conv: {config}")
    elif op_type == "MatMul":
      n = 1
      for d in config[5][:-1]:
        n *= d
      c = f"{config[1]},{config[2]},{config[3]},{config[4]},{d}"
      if not args.full:
        if c in self.fc_configs:
          return
        else:
          self.fc_configs.add(c)
      with open(self.matmul_path, 'a') as f:
        f.write(f"{config[0]},{c}\n")
        if debug:
          print(f"MatMul: {config}")
    else:
      raise Exception("Not implemented!")

def run(graph):
  for input_ in graph.input:
    m_dict = MessageToDict(input_)
    dim_info = m_dict["type"]["tensorType"]["shape"]["dim"]  # ugly but we have to live with this when using dict
    input_shape = [d.get("dimValue") for d in dim_info]  # [4,3,384,640]

  skipped_optype = set()
  printer = OperatorPrinter(f"{args.model}_{args.n_channel}.onnx")
  for node in graph.node:
    if node.op_type == 'Conv':
      # (N, C, H, W)
      input_shape = get_arg_shape(graph, node, node.input[0])
      if input_shape is None: # first node of the graph
        assert len(graph.input) < 2 # single input
        assert graph.node[0] == node # first node
        m_dict = MessageToDict(graph.input[0])
        dim_info = m_dict["type"]["tensorType"]["shape"]["dim"]  # ugly but we have to live with this when using dict
        input_shape = [d.get("dimValue") for d in dim_info]  # [4,3,384,640]

      # (O_c, I_c, K_h, K_w)
      weight_shape = find_initializer_by_arg_name(graph, node.input[1]).dims
      # (kernel_size, stride, padding, dilation, groups)
      attributes = {e.name:(e.ints if len(e.ints) > 0 else e.i) for e in node.attribute}
      assert all(e == attributes['dilations'][0] for e in attributes['dilations'])
      assert attributes['pads'][0] == attributes['pads'][2] and attributes['pads'][1] == attributes['pads'][3]
      assert all(e == attributes['strides'][0] for e in attributes['strides'])
      attributes = {k:v for k, v in attributes.items()}
      attributes['input_shape'] = input_shape
      attributes['weight_shape'] = weight_shape
      attributes['kernel_name'] = node.name
      attributes['bias'] = len(node.input) == 3
      attributes['activation'] = activation_type(graph, node)
      print(attributes)

      if input_shape[2] == 1 and input_shape[3] == 1 and weight_shape[2] == 1 and weight_shape[3] == 1:
        printer.print("MatMul", [node.name, weight_shape[0], weight_shape[1], int(len(node.input) > 2), activation_type(graph, node), input_shape])
        continue

      printer.print("Conv", attributes, debug=True)
    elif node.op_type == 'MatMul':
      input_shape = get_arg_shape(graph, node, node.input[0])
      weight = find_initializer_by_arg_name(graph, node.input[1])
      if weight is None:
        continue
      weight_shape = weight.dims
      printer.print("MatMul", [node.name, weight_shape[0], weight_shape[1], 0, activation_type(graph, node), input_shape])
    elif node.op_type == 'Gemm':
      input_shape = get_arg_shape(graph, node, node.input[0])
      attributes = {e.name:e for e in node.attribute}
      if np.isclose(attributes['alpha'].f, 1.0) and np.isclose(attributes['beta'].f, 1.0) and attributes.get('transA', 0) == 0:
        weight = find_initializer_by_arg_name(graph, node.input[1])
        if weight is None:
          print(f'Skipped Gemm: {node.input[1]}')
          continue
        weight_shape = weight.dims
        printer.print("MatMul", [node.name, weight_shape[0], weight_shape[1], int(len(node.input) > 2), activation_type(graph, node), input_shape])
      else:
        skipped_optype.add(node.op_type)
    else:
      skipped_optype.add(node.op_type)

  print(f"Skipped: {skipped_optype}")

from pim.transform import InputSplit

onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
ratio = args.split_ratio/100

onnx_model = InputSplit(ratio).transform(onnx_model)
onnx.checker.check_model(onnx_model)
onnx.save(onnx_model, f"{args.model}_{args.n_channel}_transformed.onnx")
onnx_model = onnx.load(f"{args.model}_{args.n_channel}_transformed.onnx")
run(onnx_model.graph)
