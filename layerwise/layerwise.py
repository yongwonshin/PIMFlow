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
parser.add_argument("--ic", type=int, required=True)
parser.add_argument("--oc", type=int, required=True)
parser.add_argument("--h", type=int, required=True)
parser.add_argument("--w", type=int, required=True)
parser.add_argument("--kh", type=int, required=True)
parser.add_argument("--kw", type=int, required=True)
parser.add_argument("--stride", type=int)
parser.add_argument("--ph", type=int, required=True)
parser.add_argument("--pw", type=int, required=True)
parser.add_argument("--dilate", type=int, required=True)
parser.add_argument("--g", type=int, required=True)
parser.add_argument("--b", action="store_true", default=False)
parser.add_argument("--dev", type=int, required=True)
parser.add_argument("--activation", required=True)
args = parser.parse_args()

marker = load(name="marker", sources = ["/root/PIMFlow/marker/marker_cuda.cpp", "/root/PIMFlow/marker/marker_cuda_kernel.cu"])

class Net(torch.nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.conv = torch.nn.Conv2d(args.ic, args.oc, (args.kh, args.kw), args.stride, (args.ph, args.pw), args.dilate, args.g, args.b)

  def forward(self, x):
    x = self.conv(x)
    return x

class NetElemwiseAct(torch.nn.Module):
  def __init__(self, type):
    super(NetElemwiseAct, self).__init__()
    self.conv = torch.nn.Conv2d(args.ic, args.oc, (args.kh, args.kw), args.stride, (args.ph, args.pw), args.dilate, args.g, args.b)
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
    self.conv = torch.nn.Conv2d(args.ic, args.oc, (args.kh, args.kw), args.stride, (args.ph, args.pw), args.dilate, args.g, args.b)

  def forward(self, x):
    x = self.conv(x)
    x2 = torch.sigmoid(x)
    x = torch.mul(x, x2)
    return x

class NetHardSiLU(torch.nn.Module):
  def __init__(self):
    super(NetHardSiLU, self).__init__()
    self.conv = torch.nn.Conv2d(args.ic, args.oc, (args.kh, args.kw), args.stride, (args.ph, args.pw), args.dilate, args.g, args.b)
    self.hardsigmoid = torch.nn.Hardsigmoid()

  def forward(self, x):
    x = self.conv(x)
    x2 = self.hardsigmoid(x)
    x = torch.mul(x, x2)
    return x

class NetClip(torch.nn.Module):
  def __init__(self, min, max):
    super(NetClip, self).__init__()
    self.conv = torch.nn.Conv2d(args.ic, args.oc, (args.kh, args.kw), args.stride, (args.ph, args.pw), args.dilate, args.g, args.b)
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

print(torch.cuda.device_count())
model.cuda()
model.eval()
model.half()
x = torch.randn(1, args.ic, args.h, args.w).cuda().half()

torch_out = model(x)

# Export the model
torch.onnx.export(model,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  f"layer-{os.getpid()}.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=11,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  # training=TrainingMode.TRAINING,
                  input_names = ['input'],   # the model's input names
                  output_names = ['output']) # the model's output names
                  # dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                  #               'output' : {0 : 'batch_size'}})

onnx_model = onnx.load(f"layer-{os.getpid()}.onnx")
onnx.checker.check_model(onnx_model)

def to_numpy(tensor):
  return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

shape_dict = {"input": x.shape}
mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
desired_layouts = {
  'nn.conv2d': ['NHWC', 'OHWI'],
  'nn.max_pool2d': ['NHWC'],
  'nn.global_avg_pool2d': ['NHWC'],
}
seq = tvm.transform.Sequential([relay.transform.RemoveUnusedFunctions(),
                                relay.transform.ConvertLayout(desired_layouts),
                                relay.transform.FoldConstant()])
with tvm.transform.PassContext(opt_level=3):
  mod = seq(mod)

# debug
# print(mod)

from tvm.relay.op.contrib.pim import partition_for_pim
from tvm.contrib.pim import build_pim_kernels
mod = partition_for_pim(mod)

# debug
# print(mod)

target = "cuda -libs=cudnn,cublas"
with tvm.transform.PassContext(opt_level=2):
  lib = relay.build(mod, target, params=params)
os.system(f"mkdir -p tmp-{os.getpid()}")
lib = build_pim_kernels(lib, f"./tmp-{os.getpid()}", f"compile-{os.getpid()}.so")
dev = tvm.cuda(0)
module = graph_executor.GraphModule(lib["default"](dev))
module.set_input(**{"input" : tvm.nd.array(to_numpy(x).astype("float16"), device=dev)})

for i in range(3):
  marker.forward(True)
  module.run()
  marker.forward(False)

print("FINISH!!!")
