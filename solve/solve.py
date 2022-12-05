import pandas as pd
import math
import os
import csv
import argparse
from pim.util import MODEL_LIST
class Range(object):
  def __init__(self, start, end):
    self.start = start
    self.end = end
  def __eq__(self, other):
    return self.start <= other <= self.end

parser = argparse.ArgumentParser()
parser.add_argument("--model", help="model", choices=MODEL_LIST, required=True)
parser.add_argument("--pipeline", choices=["none", "1", "2", "3", "all"], required=True)
parser.add_argument("--n_channel", type=int, default=16)
args = parser.parse_args()

baseline = pd.read_csv(f'../pipeline/baseline_end_to_end_{args.model}_{args.n_channel}.csv', delimiter=',')
gpu = pd.read_csv(f'../pipeline/gpu_end_to_end_{args.model}_{args.n_channel}.csv', delimiter=',')
split = pd.read_csv(f'../pipeline/max_performance_end_to_end_{args.model}_{args.n_channel}.csv', delimiter=',')

pipeline1 = None
pipeline2 = None
pipeline3 = None
if os.path.exists(f'../pipeline/{args.model}_pipeline1_{args.n_channel}.csv'):
  pipeline1 = pd.read_csv(f'../pipeline/{args.model}_pipeline1_{args.n_channel}.csv', delimiter=',')
if os.path.exists(f'../pipeline/{args.model}_pipeline2_{args.n_channel}.csv'):
  pipeline2 = pd.read_csv(f'../pipeline/{args.model}_pipeline2_{args.n_channel}.csv', delimiter=',')
if os.path.exists(f'../pipeline/{args.model}_pipeline3_{args.n_channel}.csv'):
  pipeline3 = pd.read_csv(f'../pipeline/{args.model}_pipeline3_{args.n_channel}.csv', delimiter=',')


pipeline1_onnx = None
pipeline2_onnx = None
pipeline3_onnx = None
if os.path.exists(f'../pipeline/{args.model}_pipelined1_{args.n_channel}.onnx_conv.csv'):
  pipeline1_onnx = pd.read_csv(f'../pipeline/{args.model}_pipelined1_{args.n_channel}.onnx_conv.csv', delimiter=',',header=None)
if os.path.exists(f'../pipeline/{args.model}_pipelined2_{args.n_channel}.onnx_conv.csv'):
  pipeline2_onnx = pd.read_csv(f'../pipeline/{args.model}_pipelined2_{args.n_channel}.onnx_conv.csv', delimiter=',',header=None)
if os.path.exists(f'../pipeline/{args.model}_pipelined3_{args.n_channel}.onnx_conv.csv'):
  pipeline3_onnx = pd.read_csv(f'../pipeline/{args.model}_pipelined3_{args.n_channel}.onnx_conv.csv', delimiter=',',header=None)


assert len(list(baseline.values)) == len(list(split.values))

N = len(list(baseline.values))

dp_b = [[float("infinity") for _ in range(N+1)] for _ in range(N+1)]
dp_s = [[float("infinity") for _ in range(N+1)] for _ in range(N+1)]
dp_ws = [[-1 for _ in range(N+1)] for _ in range(N+1)]
pipeline_cycles = [[None for _ in range(N+1)] for _ in range(N+1)]
trace_name = [[None for _ in range(N+1)] for _ in range(N+1)]
pipeline_type = [[None for _ in range(N+1)] for _ in range(N+1)]
optimal_name = []

baseline_cycle = 0
newton_cycle = 0
for i, row in enumerate(baseline.values):
  row = list(row)
  baseline_cycle += float(row[-5])

for i, row in enumerate(gpu.values):
  row = list(row)
  cycle = min(float(row[-5]), float(row[-4]))
  dp_b[i+1][1] = cycle
  newton_cycle += cycle

split_cycle = 0
for i, row in enumerate(split.values):
  row = list(row)
  cycle = float(row[-3])
  if math.isclose(float(row[-1]), 0):
    cycle = dp_b[i+1][1]
  dp_s[i+1][1] = cycle
  dp_ws[i+1][1] = cycle
  optimal_name.append([row[0],"split",row[-2],row[-6]])
  split_cycle += cycle


pipelines = set()
worst_pipelines = set()
valids = set()

# table for storing minimum runtime from jth node for 'i' number of nodes
idx = 0
idx_v = 0
while True:
  if args.pipeline not in ["1", "all"] or pipeline1 is None:
    break
  cycle = 0
  rows = list(pipeline1.values)
  row = list(rows[idx])
  if "pim" in row[0]:
    cycle += float(rows[idx][-1])
    cycle += max(float(rows[idx+1][-1]), float(rows[idx+1][-2]))
    cycle += float(rows[idx+2][-2])
    dp_b[idx_v+1][2] = cycle
    dp_s[idx_v+1][2] = cycle
    dp_ws[idx_v+1][2] = cycle
    pipeline_cycles[idx_v+1][2] = ([float(rows[idx][-1]), max(float(rows[idx+1][-1]), float(rows[idx+1][-2])), float(rows[idx+2][-2])], 1)
    trace_name[idx_v+1][2] = str(rows[idx+2][0])
    pipeline_type[idx_v+1][2] = 1
    valids.add((idx_v+1, 2))
    idx += 3
    idx_v += 2
  else:
    idx += 1
    idx_v += 1

  if idx_v >= N:
    break

# table for storing minimum runtime from jth node for 'i' number of nodes
idx = 0
idx_v = 0
while True:
  if args.pipeline not in ["2", "all"] or pipeline2 is None:
    break
  cycle = 0
  rows = list(pipeline2.values)
  row = list(rows[idx])
  if "added" in row[0]:
    cycle += float(rows[idx][-2])
    cycle += max(float(rows[idx+1][-1]), float(rows[idx+1][-2]))
    cycle += float(rows[idx+2][-1])
    dp_b[idx_v+1][2] = cycle
    dp_s[idx_v+1][2] = cycle
    dp_ws[idx_v+1][2] = cycle
    pipeline_cycles[idx_v+1][2] = ([float(rows[idx][-2]), max(float(rows[idx+1][-1]), float(rows[idx+1][-2])), float(rows[idx+2][-1])], 2)
    trace_name[idx_v+1][2] = str(rows[idx][0])
    pipeline_type[idx_v+1][2] = 2
    valids.add((idx_v+1, 2))
    idx += 3
    idx_v += 2
  else:
    idx += 1
    idx_v += 1

  if idx_v >= N:
    break

# table for storing minimum runtime from jth node for 'i' number of nodes
idx = 0
idx_v = 0
while True:
  if args.pipeline not in ["3", "all"] or pipeline3 is None:
    break
  cycle = 0
  rows = list(pipeline3.values)
  row = list(rows[idx])
  if "pim" in row[0]:
    cycle += float(rows[idx][-1])
    cycle += max(float(rows[idx+1][-1]), float(rows[idx+1][-2]))
    cycle += max(float(rows[idx+2][-1]), float(rows[idx+2][-2]))
    cycle += float(rows[idx+3][-1])
    dp_b[idx_v+1][3] = cycle
    dp_s[idx_v+1][3] = cycle
    dp_ws[idx_v+1][3] = cycle
    pipeline_cycles[idx_v+1][3] = ([float(rows[idx][-1]), max(float(rows[idx+1][-1]), float(rows[idx+1][-2])), max(float(rows[idx+2][-1]), float(rows[idx+2][-2])), float(rows[idx+3][-1])], 3)
    trace_name[idx_v+1][3] = str(rows[idx+2][0])
    pipeline_type[idx_v+1][3] = 3
    valids.add((idx_v+1, 3))
    idx += 4
    idx_v += 3
  else:
    idx += 1
    idx_v += 1

  if idx_v >= N:
    break


# solve
eaten = 0
for l in range(1, N+1):
  for i in range(1, N+1):
    for k in range(1, l):
      if i + k > N:
        continue
      if l == 2 or l == 3:
        if dp_s[i][l] < dp_s[i][k] + dp_s[i+k][l-k] - 1:
          if (i, k) in pipelines:
            pipelines.remove((i, k))
            eaten += 1
          if (i+k, l-k) in pipelines:
            pipelines.remove((i+k, l-k))
            eaten += 1
          if (i, l) in valids:
            pipelines.add((i, l))
        if dp_ws[i][l] > dp_ws[i][k] + dp_ws[i+k][l-k] + 1:
          if (i, k) in worst_pipelines:
            worst_pipelines.remove((i, k))
            eaten += 1
          if (i+k, l-k) in worst_pipelines:
            worst_pipelines.remove((i+k, l-k))
            eaten += 1
          if (i, l) in valids:
            worst_pipelines.add((i, l))
      dp_b[i][l] = min(dp_b[i][l], dp_b[i][k] + dp_b[i+k][l-k])
      dp_s[i][l] = min(dp_s[i][l], dp_s[i][k] + dp_s[i+k][l-k])
      dp_ws[i][l] = max(dp_ws[i][l], dp_ws[i][k] + dp_ws[i+k][l-k])



def pipeline_type_to(t):
  if t == 2:
    return "g"
  else:
    return "p"

removes = []
for p in pipelines:
  i, l = p
  b = [dp_s[i+j][1] for j in range(l)]
  if abs(1 - dp_s[i][l] / sum(b)) < 0.05:
    removes.append(p)
for r in removes:
  pipelines.remove(r)

removes = []
for p in worst_pipelines:
  i, l = p
  b = [dp_s[i+j][1] for j in range(l)]
  if abs(1 - dp_ws[i][l] / sum(b)) < 0.30:
    removes.append(p)
for r in removes:
  worst_pipelines.remove(r)

for p in pipelines:
  i, l = p
  b = [dp_s[i+j][1] for j in range(l)]


for p in worst_pipelines:
  i, l = p
  b = [dp_ws[i+j][1] for j in range(l)]


for p in pipelines:
  i, l = p
  b = [dp_s[i+j][1] for j in range(l)]
  optimal_name[i] = [optimal_name[i][0],"pipeline",pipeline_type[i][l],trace_name[i][l]]


for p in worst_pipelines:
  i, l = p
  b = [dp_ws[i+j][1] for j in range(l)]


# final

os.system(f'mkdir /root/PIMFlow/{args.model}')
OPTIMAL=[]


for i, k in enumerate(optimal_name):
  if optimal_name[i][3] != "pim":
    if k[2] == 1:
      optimal_name[i-1][1] = "pipeline"
      optimal_name[i-1][2] = 1
      optimal_name[i-1][3] = "pim"
    elif k[2] == 2:
      optimal_name[i+1][1] = "pipeline"
      optimal_name[i+1][2] = 2
      optimal_name[i+1][3] = "pim"
    elif k[2] == 3:
      optimal_name[i-1][1] = "pipeline"
      optimal_name[i-1][2] = 3
      optimal_name[i-1][3] = "pim"
      optimal_name[i+1][1] = "pipeline"
      optimal_name[i+1][2] = 3
      optimal_name[i+1][3] = "pim"


if pipeline1_onnx is not None:
  pipeline1_onnx=list(pipeline1_onnx.values)
else:
  pipeline1_onnx = []
if pipeline2_onnx is not None:
  pipeline2_onnx=list(pipeline2_onnx.values)
else:
  pipeline2_onnx = []
if pipeline3_onnx is not None:
  pipeline3_onnx=list(pipeline3_onnx.values)
else:
  pipeline3_onnx = []
for i, k in enumerate(optimal_name):
  if k[1] == "split":
    if k[2] != 0:
      os.system(f'cp -r /root/PIMFlow/layerwise/result_simulate/{args.model}/{k[2]}_16/traces-{k[3]} /root/PIMFlow/{args.model}/trace-{k[0]}')
    optimal_name[i][3] = f'trace-{k[0]}'
    OPTIMAL.append(k)

  elif k[1] == "pipeline" and k[3] != "pim":
    if k[2] == 1:
      for j, row in enumerate(pipeline1_onnx):
        if row[0]== k[3]:
          optimal_name[i].append(pipeline1_onnx[j-1][0])
          os.system(f'cp -r /root/PIMFlow/pipeline/result_simulate/{args.model}/{k[2]}_16/traces-{k[3]} /root/PIMFlow/{args.model}/trace-{k[0]}_2')
          os.system(f'cp -r /root/PIMFlow/pipeline/result_simulate/{args.model}/{k[2]}_16/traces-{k[4]} /root/PIMFlow/{args.model}/trace-{k[0]}_1')
          optimal_name[i][3] = f'trace-{k[0]}_1'
          optimal_name[i][4] = f'trace-{k[0]}_2'
          optimal_name[i][3] = f'trace-{k[0]}_1'
          optimal_name[i][4] = f'trace-{k[0]}_2'
          optimal_name[i].insert(0,optimal_name[i-1][0])
    elif k[2] == 2:
      for j, row in enumerate(pipeline2_onnx):
        if row[0]== k[3]:
          optimal_name[i].append(pipeline2_onnx[j+1][0])
          os.system(f'cp -r /root/PIMFlow/pipeline/result_simulate/{args.model}/{k[2]}_16/traces-{k[3]} /root/PIMFlow/{args.model}/trace-{k[0]}_1')
          os.system(f'cp -r /root/PIMFlow/pipeline/result_simulate/{args.model}/{k[2]}_16/traces-{k[4]} /root/PIMFlow/{args.model}/trace-{k[0]}_2')
          optimal_name[i][3] = f'trace-{k[0]}_1'
          optimal_name[i][4] = f'trace-{k[0]}_2'
          optimal_name[i].insert(1,optimal_name[i+1][0])
    elif k[2] == 3:
      for j, row in enumerate(pipeline3_onnx):
        if row[0]== k[3]:
          optimal_name[i].append(pipeline3_onnx[j-1][0])
          os.system(f'cp -r /root/PIMFlow/pipeline/result_simulate/{args.model}/{k[2]}_16/traces-{k[3]} /root/PIMFlow/{args.model}/trace-{k[0]}_2')
          os.system(f'cp -r /root/PIMFlow/pipeline/result_simulate/{args.model}/{k[2]}_16/traces-{k[4]} /root/PIMFlow/{args.model}/trace-{k[0]}_1')
          optimal_name[i][3] = f'trace-{k[0]}_1'
          optimal_name[i][4] = f'trace-{k[0]}_2'
          optimal_name[i].insert(0,optimal_name[i-1][0])
          optimal_name[i].insert(2,optimal_name[i+1][0])
    OPTIMAL.append(k)


with open(f'solve_{args.model}.csv', 'w',newline='') as f:
  write = csv.writer(f)
  write.writerows(OPTIMAL)
os.system(f'mv solve_{args.model}.csv /root/PIMFlow/{args.model}/')
for i in OPTIMAL:
  print(i)
