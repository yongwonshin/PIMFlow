import csv
import pandas as pd

import argparse
import os
from pim.util import MODEL_LIST
parser = argparse.ArgumentParser()
parser.add_argument("--model", help="model", choices=MODEL_LIST, required=True)
parser.add_argument("--n_channel", type=int, default=16)
args = parser.parse_args()

def process(model):
  end_to_end = pd.read_csv(f'{model}_{args.n_channel}.onnx_conv.csv', delimiter=',',header=None)
  END_base = [list(row) for row in end_to_end.values]
  END_gpu = [list(row) for row in end_to_end.values]
  END_max = [list(row) for row in end_to_end.values]

  baseline = pd.read_csv(f'{model}_split100-baseline.csv', delimiter=',')
  gpu = pd.read_csv(f'{model}_split100_{args.n_channel}.csv', delimiter=',')
  head = ["kernel_name","N","I_c","H","W","O_c","kernel_size","pads","strides","group","dilations","bias","activation","GPU cycles","PIM cycles","TOTAL_cycle","RATIO","SPEED_UP"]

  newton = pd.read_csv(f'../layerwise/newton_performance_{model}_{args.n_channel}.csv', delimiter=',')

  # head.append("TOTAL_cycel")
  # head.append("RATIO")
  # head.append("SPEED_UP")
  # head.insert(1,'N')
  act = 0
  if 'activation' in baseline:
    act = 1


  BASE = [list(row) for row in baseline.values]
  GPU = [list(row) for row in gpu.values]
  NEWTON = [list(row) for row in newton.values]
  DIC_GPU_base={}
  DIC_GPU_only={}
  DIC_PIM_base={}
  for i in range(len(BASE)):
    key = str(BASE[i][1]) + str(BASE[i][2]) + str(BASE[i][4]) + str(BASE[i][5]) + str(BASE[i][6][3]) + str(BASE[i][7]) + str(BASE[i][8])
    DIC_GPU_base[key] = BASE[i][11+act]
    DIC_GPU_only[key] = GPU[i][11+act]
    DIC_PIM_base[key] = NEWTON[i][12+act]

  DIC_GPU_max={}
  DIC_PIM_max={}
  DIC_TOT_max={}
  DIC_RATIO_max={}
  DIC_SPEED_max={}
  max_ = pd.read_csv(f'max_performance_{model}_{args.n_channel}.csv', delimiter=',')
  MAX = [list(row) for row in max_.values]
  for i in range(len(MAX)):
    key = str(MAX[i][1]) + str(MAX[i][2]) + str(MAX[i][4]) + str(MAX[i][5]) + str(MAX[i][6][3]) + str(MAX[i][7]) + str(MAX[i][8])
    DIC_GPU_max[key] = MAX[i][11+act]
    DIC_PIM_max[key] = MAX[i][12+act]
    DIC_TOT_max[key] = MAX[i][13+act]
    DIC_RATIO_max[key] = MAX[i][14+act]
    DIC_SPEED_max[key] = MAX[i][15+act]

  for i in range(len(END_base)):
    key = str(END_base[i][2]) + str(END_base[i][5]) + str(END_base[i][4]) + str(END_base[i][6]) + str(END_base[i][7][3]) + str(END_base[i][8]) + str(END_base[i][9])
    END_base[i].append(DIC_GPU_base.get(key,0))
    END_base[i].append(DIC_PIM_base.get(key,0))
    END_base[i].append(min(END_base[i][-2], END_base[i][-1])) # total
    if DIC_GPU_base.get(key,0) <= DIC_PIM_base.get(key,0): # ratio
      END_base[i].append(100)
    else:
      END_base[i].append(0)
    END_base[i].append(max(END_base[i][-4] / END_base[i][-3], 1)) # speedup

  for i in range(len(END_gpu)):
    key = str(END_gpu[i][2]) + str(END_gpu[i][5]) + str(END_gpu[i][4]) + str(END_gpu[i][6]) + str(END_gpu[i][7][3]) + str(END_gpu[i][8]) + str(END_gpu[i][9])
    END_gpu[i].append(DIC_GPU_only.get(key,0))
    END_gpu[i].append(DIC_PIM_base.get(key,0))
    END_gpu[i].append(min(END_gpu[i][-2], END_gpu[i][-1])) # total
    if DIC_GPU_only.get(key,0) <= DIC_PIM_base.get(key,0): # ratio
      END_gpu[i].append(100)
    else:
      END_gpu[i].append(0)
    END_gpu[i].append(max(END_gpu[i][-4] / END_gpu[i][-3], 1)) # speedup

  for i in range(len(END_max)):
    key = str(END_max[i][2]) + str(END_max[i][5]) + str(END_max[i][4]) + str(END_max[i][6]) + str(END_max[i][7][3]) + str(END_max[i][8]) + str(END_max[i][9])
    END_max[i].append(DIC_GPU_max.get(key, 0))
    END_max[i].append(DIC_PIM_max.get(key, 0))
    END_max[i].append(DIC_TOT_max.get(key, 0))
    END_max[i].append(DIC_RATIO_max.get(key, 0))
    END_max[i].append(DIC_SPEED_max.get(key, 0))

  with open(f'max_performance_end_to_end_{model}_{args.n_channel}.csv', 'w',newline='') as f:
    write = csv.writer(f)
    write.writerow(head)
    write.writerows(END_max)

  with open(f'baseline_end_to_end_{model}_{args.n_channel}.csv', 'w',newline='') as f:
    write = csv.writer(f)
    write.writerow(head)
    write.writerows(END_base)

  with open(f'gpu_end_to_end_{model}_{args.n_channel}.csv', 'w',newline='') as f:
    write = csv.writer(f)
    write.writerow(head)
    write.writerows(END_gpu)

os.system(f"python /root/PIMFlow/layerwise/inspect_shape.py --model={args.model} --split_ratio=100 --full --n_channel={args.n_channel}")
os.system(f"cp ../layerwise/max_performance_{args.model}_{args.n_channel}.csv ./")
os.system(f"cp ../layerwise/{args.model}_split100-baseline.csv ./")
os.system(f"cp ../layerwise/{args.model}_split100_{args.n_channel}.csv ./")
# os.system(f"cp ../layerwise/{args.model}.onnx_conv.csv ./")
process(args.model)
