from numpy import clip
import torch
import argparse
import os
import onnx
import tvm
import tvm.relay as relay
from tvm.contrib import graph_executor
from torch.utils.cpp_extension import load

import argparse

class Range(object):
  def __init__(self, start, end):
    self.start = start
    self.end = end
  def __eq__(self, other):
    return self.start <= other <= self.end

parser = argparse.ArgumentParser()
parser.add_argument("--batch", type=int, default=1)
parser.add_argument("--row", type=int, required=True)
parser.add_argument("--col", type=int, required=True)
parser.add_argument("--activation", choices=["SiLU", "Sigmoid", "Relu", "Identity"], required=True)
parser.add_argument("--name", required=True)
parser.add_argument("--bias", type=int, required=True)
args = parser.parse_args()

if args.bias != 0:
  args.bias = True
else:
  args.bias = False

marker = load(name="marker", sources = ["/root/PIMFlow/marker/marker_cuda.cpp", "/root/PIMFlow/marker/marker_cuda_kernel.cu"])

class Net(torch.nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.conv = torch.nn.Linear(args.col, args.row, bias=args.bias)

  def forward(self, x):
    x = self.conv(x)
    return x

class NetElemwiseAct(torch.nn.Module):
  def __init__(self, type):
    super(NetElemwiseAct, self).__init__()
    self.conv = torch.nn.Linear(args.col, args.row, bias=args.bias)
    if type == "HardSigmoid":
      self.act = torch.nn.Hardsigmoid()
    elif type == "Sigmoid":
      self.act = torch.nn.Sigmoid()
    elif type == "Relu":
      self.act = torch.nn.ReLU()
    else:
      raise Exception(f"Unknown activation: {type}")

  def forward(self, x):
    x = self.conv(x)
    x = self.act(x)
    return x

class NetSiLU(torch.nn.Module):
  def __init__(self):
    super(NetSiLU, self).__init__()
    self.conv = torch.nn.Linear(args.col, args.row, bias=args.bias)

  def forward(self, x):
    x = self.conv(x)
    x2 = torch.sigmoid(x)
    x = torch.mul(x, x2)
    return x

class NetHardSiLU(torch.nn.Module):
  def __init__(self):
    super(NetHardSiLU, self).__init__()
    self.conv = torch.nn.Linear(args.col, args.row, bias=args.bias)
    self.hardsigmoid = torch.nn.Hardsigmoid()

  def forward(self, x):
    x = self.conv(x)
    x2 = self.hardsigmoid(x)
    x = torch.mul(x, x2)
    return x

class NetClip(torch.nn.Module):
  def __init__(self, min, max):
    super(NetClip, self).__init__()
    self.conv = torch.nn.Linear(args.col, args.row, bias=args.bias)
    self.min = min
    self.max = max

  def forward(self, x):
    x = self.conv(x)
    x = torch.clip(x, min=self.min, max=self.max)
    return x

if args.activation == "SiLU":
  model = NetSiLU()
elif args.activation == "HardSiLU":
  model = NetHardSiLU()
elif args.activation in ["HardSigmoid", "Sigmoid", "Relu"]:
  model = NetElemwiseAct(type=args.activation)
elif args.activation == "Clip":
  model = NetClip(min=0, max=6)
elif args.activation == "Identity":
  model = Net()
else:
  raise Exception("Not supported activation!")

model.cuda()
model.eval()
model.half()
x = torch.randn(args.batch, args.col).cuda().half()

torch_out = model(x)

# Export the model
torch.onnx.export(model,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  f"{args.name}-{os.getpid()}.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=11,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output']) # the model's output names

onnx_model = onnx.load(f"{args.name}-{os.getpid()}.onnx")
onnx.checker.check_model(onnx_model)

def to_numpy(tensor):
  return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

shape_dict = {"input": x.shape}
mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

# print(mod)

target = "cuda -libs=cudnn,cublas"
with tvm.transform.PassContext(opt_level=2):
  lib = relay.build(mod, target, params=params)
dev = tvm.cuda(0)
module = graph_executor.GraphModule(lib["default"](dev))
module.set_input(**{"input" : tvm.nd.array(to_numpy(x).astype("float16"), device=dev)})

for i in range(3):
  marker.forward(True)
  module.run()
  marker.forward(False)

print("FINISH!!!")
