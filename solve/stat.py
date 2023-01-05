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
parser.add_argument("--n_gwrite", type=int, default=4)
parser.add_argument("--ramulator_disable_gwrite_latency_hiding", action="store_true")
parser.add_argument("--stage", type=int, default=2)
args = parser.parse_args()

postfix = ""
if args.ramulator_disable_gwrite_latency_hiding:
  postfix = "_noopt"

baseline = pd.read_csv(f'../pipeline/baseline_end_to_end_{args.model}_{args.n_channel}_{args.n_gwrite}{postfix}.csv', delimiter=',')
gpu = pd.read_csv(f'../pipeline/gpu_end_to_end_{args.model}_{args.n_channel}_{args.n_gwrite}{postfix}.csv', delimiter=',')
split = pd.read_csv(f'../pipeline/max_performance_end_to_end_{args.model}_{args.n_channel}_{args.n_gwrite}{postfix}.csv', delimiter=',')

pipeline1 = None
pipeline2 = None
pipeline3 = None
if os.path.exists(f'../pipeline/{args.model}_pipeline1_{args.n_channel}_{args.n_gwrite}{postfix}.csv'):
  pipeline1 = pd.read_csv(f'../pipeline/{args.model}_pipeline1_{args.n_channel}_{args.n_gwrite}{postfix}.csv', delimiter=',')
if os.path.exists(f'../pipeline/{args.model}_pipeline2_{args.n_channel}_{args.n_gwrite}{postfix}.csv'):
  pipeline2 = pd.read_csv(f'../pipeline/{args.model}_pipeline2_{args.n_channel}_{args.n_gwrite}{postfix}.csv', delimiter=',')
if os.path.exists(f'../pipeline/{args.model}_pipeline3_{args.n_channel}_{args.n_gwrite}{postfix}.csv'):
  pipeline3 = pd.read_csv(f'../pipeline/{args.model}_pipeline3_{args.n_channel}_{args.n_gwrite}{postfix}.csv', delimiter=',')


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
    if args.stage == 2:
      # cycle += float(rows[idx][-1])
      # cycle += max(float(rows[idx+1][-1]), float(rows[idx+1][-2]))
      # cycle += float(rows[idx+2][-2])
      cycle += float(rows[idx+1][-1])
      cycle += max(float(rows[idx+2][-2]), float(rows[idx][-1]))
      cycle += float(rows[idx+1][-2])
      dp_b[idx_v+1][2] = cycle
      dp_s[idx_v+1][2] = cycle
      dp_ws[idx_v+1][2] = cycle
      pipeline_cycles[idx_v+1][2] = ([float(rows[idx][-1]), max(float(rows[idx+1][-1]), float(rows[idx+1][-2])), float(rows[idx+2][-2])], 1)
      trace_name[idx_v+1][2] = str(rows[idx+2][0])
      pipeline_type[idx_v+1][2] = 1
      valids.add((idx_v+1, 2))
      idx += 3
      idx_v += 2
      continue
    elif args.stage == 3:
      # TODO
      cycle += float(rows[idx][-1])
      cycle += max(float(rows[idx+2][-2]), float(rows[idx+1][-1]))
      cycle += max(float(rows[idx+3][-2]), float(rows[idx+2][-1]))
      cycle += float(rows[idx+4][-2])
      dp_b[idx_v+1][2] = cycle
      dp_s[idx_v+1][2] = cycle
      dp_ws[idx_v+1][2] = cycle
      pipeline_cycles[idx_v+1][2] = (cycle, 1)
      valids.add((idx_v+1, 2))
      idx += 5
      idx_v += 2
      continue
    elif args.stage == 4:
      # TODO
      cycle += float(rows[idx][-1])
      cycle += max(float(rows[idx+3][-2]), float(rows[idx+1][-1]))
      cycle += max(float(rows[idx+4][-2]), float(rows[idx+2][-1]))
      cycle += max(float(rows[idx+5][-2]), float(rows[idx+3][-1]))
      cycle += float(rows[idx+6][-2])
      dp_b[idx_v+1][2] = cycle
      dp_s[idx_v+1][2] = cycle
      dp_ws[idx_v+1][2] = cycle
      pipeline_cycles[idx_v+1][2] = (cycle, 1)
      valids.add((idx_v+1, 2))
      idx += 7
      idx_v += 2
      continue
    elif args.stage == 5:
      # TODO
      cycle += float(rows[idx][-1])
      cycle += max(float(rows[idx+4][-2]), float(rows[idx+1][-1]))
      cycle += max(float(rows[idx+5][-2]), float(rows[idx+2][-1]))
      cycle += max(float(rows[idx+6][-2]), float(rows[idx+3][-1]))
      cycle += max(float(rows[idx+7][-2]), float(rows[idx+4][-1]))
      cycle += float(rows[idx+8][-2])
      dp_b[idx_v+1][2] = cycle
      dp_s[idx_v+1][2] = cycle
      dp_ws[idx_v+1][2] = cycle
      pipeline_cycles[idx_v+1][2] = (cycle, 1)
      valids.add((idx_v+1, 2))
      idx += 9
      idx_v += 2
      continue

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
    if args.stage == 2:
      # cycle += float(rows[idx][-2])
      # cycle += max(float(rows[idx+1][-1]), float(rows[idx+1][-2]))
      # cycle += float(rows[idx+2][-1])
      cycle += float(rows[idx+1][-2])
      cycle += max(float(rows[idx][-2]), float(rows[idx+2][-1]))
      cycle += float(rows[idx+1][-1])
      dp_b[idx_v+1][2] = cycle
      dp_s[idx_v+1][2] = cycle
      dp_ws[idx_v+1][2] = cycle
      pipeline_cycles[idx_v+1][2] = ([float(rows[idx][-2]), max(float(rows[idx+1][-1]), float(rows[idx+1][-2])), float(rows[idx+2][-1])], 2)
      trace_name[idx_v+1][2] = str(rows[idx][0])
      pipeline_type[idx_v+1][2] = 2
      valids.add((idx_v+1, 2))
      idx += 3
      idx_v += 2
      continue
    elif args.stage == 3:
      # TODO
      cycle += float(rows[idx][-2])
      cycle += max(float(rows[idx+1][-2]), float(rows[idx+2][-1]))
      cycle += max(float(rows[idx+2][-2]), float(rows[idx+3][-1]))
      cycle += float(rows[idx+4][-1])
      dp_b[idx_v+1][2] = cycle
      dp_s[idx_v+1][2] = cycle
      dp_ws[idx_v+1][2] = cycle
      pipeline_cycles[idx_v+1][2] = (cycle, 2)
      valids.add((idx_v+1, 2))
      idx += 5
      idx_v += 2
      continue
    elif args.stage == 4:
      # TODO
      cycle += float(rows[idx][-2])
      cycle += max(float(rows[idx+1][-2]), float(rows[idx+3][-1]))
      cycle += max(float(rows[idx+2][-2]), float(rows[idx+4][-1]))
      cycle += max(float(rows[idx+3][-2]), float(rows[idx+5][-1]))
      cycle += float(rows[idx+6][-1])
      dp_b[idx_v+1][2] = cycle
      dp_s[idx_v+1][2] = cycle
      dp_ws[idx_v+1][2] = cycle
      pipeline_cycles[idx_v+1][2] = (cycle, 2)
      valids.add((idx_v+1, 2))
      idx += 7
      idx_v += 2
      continue
    elif args.stage == 5:
      # TODO
      cycle += float(rows[idx][-2])
      cycle += max(float(rows[idx+1][-2]), float(rows[idx+4][-1]))
      cycle += max(float(rows[idx+2][-2]), float(rows[idx+5][-1]))
      cycle += max(float(rows[idx+3][-2]), float(rows[idx+6][-1]))
      cycle += max(float(rows[idx+4][-2]), float(rows[idx+7][-1]))
      cycle += float(rows[idx+8][-1])
      dp_b[idx_v+1][2] = cycle
      dp_s[idx_v+1][2] = cycle
      dp_ws[idx_v+1][2] = cycle
      pipeline_cycles[idx_v+1][2] = (cycle, 2)
      valids.add((idx_v+1, 2))
      idx += 9
      idx_v += 2
      continue

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
    if args.stage == 2:
      # cycle += float(rows[idx][-1])
      # cycle += max(float(rows[idx+1][-1]), float(rows[idx+1][-2]))
      # cycle += max(float(rows[idx+2][-1]), float(rows[idx+2][-2]))
      # cycle += float(rows[idx+3][-1])
      cycle += float(rows[idx+1][-1])
      cycle += max(float(rows[idx+2][-2]), float(rows[idx][-1]))
      cycle += max(float(rows[idx+1][-2]), float(rows[idx+3][-1]))
      cycle += float(rows[idx+2][-1])
      dp_b[idx_v+1][3] = cycle
      dp_s[idx_v+1][3] = cycle
      dp_ws[idx_v+1][3] = cycle
      pipeline_cycles[idx_v+1][3] = ([float(rows[idx][-1]), max(float(rows[idx+1][-1]), float(rows[idx+1][-2])), max(float(rows[idx+2][-1]), float(rows[idx+2][-2])), float(rows[idx+3][-1])], 3)
      trace_name[idx_v+1][3] = str(rows[idx+2][0])
      pipeline_type[idx_v+1][3] = 3
      valids.add((idx_v+1, 3))
      idx += 4
      idx_v += 3
      continue
    elif args.stage == 3:
      # TODO
      if float(rows[idx][-1]) < float(rows[idx+6][-1]):
        cycle += float(rows[idx][-1])
        cycle += float(rows[idx+1][-1])
        cycle += max(float(rows[idx+2][-2]), float(rows[idx+2][-1]))
        cycle += max(float(rows[idx+3][-2]), float(rows[idx+4][-1]))
        cycle += max(float(rows[idx+4][-2]), float(rows[idx+5][-1]))
        cycle += float(rows[idx+6][-1])
      else:
        cycle += float(rows[idx][-1])
        cycle += max(float(rows[idx+2][-2]), float(rows[idx+1][-1]))
        cycle += max(float(rows[idx+3][-2]), float(rows[idx+2][-1]))
        cycle += max(float(rows[idx+4][-2]), float(rows[idx+4][-1]))
        cycle += float(rows[idx+5][-1])
        cycle += float(rows[idx+6][-1])
      dp_b[idx_v+1][3] = cycle
      dp_s[idx_v+1][3] = cycle
      dp_ws[idx_v+1][3] = cycle
      pipeline_cycles[idx_v+1][3] = (cycle, 3)
      valids.add((idx_v+1, 3))
      idx += 7
      idx_v += 3
      continue
    elif args.stage == 4:
      # TODO
      cycle += float(rows[idx][-1])
      cycle += float(rows[idx+1][-1])
      cycle += max(float(rows[idx+3][-2]), float(rows[idx+2][-1]))
      cycle += max(float(rows[idx+4][-2]), float(rows[idx+3][-1]))
      cycle += max(float(rows[idx+5][-2]), float(rows[idx+6][-1]))
      cycle += max(float(rows[idx+6][-2]), float(rows[idx+7][-1]))
      cycle += float(rows[idx+8][-1])
      cycle += float(rows[idx+9][-1])
      dp_b[idx_v+1][3] = cycle
      dp_s[idx_v+1][3] = cycle
      dp_ws[idx_v+1][3] = cycle
      pipeline_cycles[idx_v+1][3] = (cycle, 3)
      valids.add((idx_v+1, 3))
      idx += 10
      idx_v += 3
      continue
    elif args.stage == 5:
      # TODO
      if float(rows[idx][-1]) < float(rows[idx+12][-1]):
        cycle += float(rows[idx][-1])
        cycle += float(rows[idx+1][-1])
        cycle += float(rows[idx+2][-1])
        cycle += max(float(rows[idx+4][-2]), float(rows[idx+3][-1]))
        cycle += max(float(rows[idx+5][-2]), float(rows[idx+4][-1]))
        cycle += max(float(rows[idx+6][-2]), float(rows[idx+8][-1]))
        cycle += max(float(rows[idx+7][-2]), float(rows[idx+9][-1]))
        cycle += max(float(rows[idx+8][-2]), float(rows[idx+10][-1]))
        cycle += float(rows[idx+11][-1])
        cycle += float(rows[idx+12][-1])
      else:
        cycle += float(rows[idx][-1])
        cycle += float(rows[idx+1][-1])
        cycle += max(float(rows[idx+4][-2]), float(rows[idx+2][-1]))
        cycle += max(float(rows[idx+5][-2]), float(rows[idx+3][-1]))
        cycle += max(float(rows[idx+6][-2]), float(rows[idx+4][-1]))
        cycle += max(float(rows[idx+7][-2]), float(rows[idx+8][-1]))
        cycle += max(float(rows[idx+8][-2]), float(rows[idx+9][-1]))
        cycle += float(rows[idx+10][-1])
        cycle += float(rows[idx+11][-1])
        cycle += float(rows[idx+12][-1])
      dp_b[idx_v+1][3] = cycle
      dp_s[idx_v+1][3] = cycle
      dp_ws[idx_v+1][3] = cycle
      pipeline_cycles[idx_v+1][3] = (cycle, 3)
      valids.add((idx_v+1, 3))
      idx += 13
      idx_v += 3
      continue

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
  if abs(1 - dp_s[i][l] / sum(b)) < 0.10:
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
  #print(f"GOOD PIPELINE!: {p} (pipeline: {pipeline_cycles[i][l]}) (original: {b}) ({dp_s[i][l] - sum(b)})")


for p in worst_pipelines:
  i, l = p
  b = [dp_ws[i+j][1] for j in range(l)]
  #print(f"BAD PIPELINE!: {p} (pipeline: {pipeline_cycles[i][l]}) (original: {b}) ({dp_s[i][l] - sum(b)})")


for p in pipelines:
  i, l = p
  b = [dp_s[i+j][1] for j in range(l)]
  optimal_name[i] = [optimal_name[i][0],"pipeline",pipeline_type[i][l],trace_name[i][l]]


for p in worst_pipelines:
  i, l = p
  b = [dp_ws[i+j][1] for j in range(l)]

print(f"=== N_CHANNEL: {args.n_channel}, N_GWRITE: {args.n_gwrite}, ramulator_disable_gwrite_latency_hiding: {args.ramulator_disable_gwrite_latency_hiding} ===")
print(f"newton++ (vs baseline): {round(baseline_cycle / newton_cycle, 3)} ({newton_cycle - baseline_cycle})")
print(f"pipeline (vs baseline): {round(baseline_cycle / dp_b[1][N], 3)} ({dp_b[1][N] - baseline_cycle})")
print(f"split (vs baseline): {round(baseline_cycle / split_cycle, 3)} ({split_cycle - baseline_cycle})")
print(f"all (vs baseline): {round(baseline_cycle / dp_s[1][N], 3)} ({dp_s[1][N] - baseline_cycle})")
print("====================\n")
