import csv
import pandas as pd
import copy
from pim.util import MODEL_LIST
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", help="model", choices=MODEL_LIST, required=True)
parser.add_argument("--n_channel", type=int, default=16)
parser.add_argument("--n_gwrite", type=int, default=4)
parser.add_argument("--ramulator_disable_gwrite_latency_hiding", action="store_true")
args = parser.parse_args()

postfix = ""
if args.ramulator_disable_gwrite_latency_hiding:
    postfix = "_noopt"

TERM = "split"
OFFSET = 0
if args.model in ['bert-large-1x3', 'bert-large-1x32', 'bert-large-1x64', 'bert-large-1x128', 'bert-base-1x3', 'bert-base-1x32', 'bert-base-1x64', 'bert-base-1x128']:
    TERM = "matmul"
    OFFSET = -7

def process(model):
    baseline = pd.read_csv(f'{model}_{TERM}100-baseline.csv', delimiter=',')
    BASE = [list(row) for row in baseline.values]

    if args.n_channel == 32:
        newton = pd.read_csv(f'{model}_{TERM}100_32_{args.n_gwrite}{postfix}.csv', delimiter=',')
        NEWTON = [list(row) for row in newton.values]
    else:
        newton = pd.read_csv(f'{model}_{TERM}0_{args.n_channel}_{args.n_gwrite}{postfix}.csv', delimiter=',')
        NEWTON = [list(row) for row in newton.values]

    Top = pd.read_csv(f'{model}_{TERM}100_{args.n_channel}_{args.n_gwrite}{postfix}.csv', delimiter=',')
    head = list(Top.columns)
    head.append("TOTAL_cycle")
    head.append("RATIO")
    head.append("SPEED_UP")
    MAX = [list(row) for row in Top.values]
    GPU = copy.deepcopy(MAX)

    assert len(BASE) == len(NEWTON) and len(NEWTON) == len(MAX)
    columns_len = len(MAX)

    for line in range(columns_len):
        MAX[line].append(max(MAX[line][12+OFFSET], MAX[line][13+OFFSET]))
        MAX[line].append('100')
        MAX[line].append(BASE[line][12+OFFSET] / MAX[line][14+OFFSET])

    if args.n_channel < 32:
        for i in range(0,91,10):
            print(i)
            f = pd.read_csv(f'{model}_{TERM}{i}_{args.n_channel}_{args.n_gwrite}{postfix}.csv',delimiter=',')
            tmp = [list(row) for row in f.values]
            for line in range(columns_len):
                tmp[line].append(max(tmp[line][12+OFFSET],tmp[line][13+OFFSET]))
                tmp[line].append(i)
                if tmp[line][14+OFFSET]:
                    tmp[line].append(BASE[line][12+OFFSET] / tmp[line][14+OFFSET])
            for line in range(columns_len):
                if float(tmp[line][14+OFFSET]) < float(MAX[line][14+OFFSET]):
                    MAX[line] = tmp[line]

    for line in range(columns_len):
        NEWTON[line][12+OFFSET] = BASE[line][12+OFFSET]
        NEWTON[line].append(MAX[line][14+OFFSET])
        NEWTON[line].append(0)
        NEWTON[line].append(0)
        NEWTON[line].append(0)
        if NEWTON[line][13+OFFSET] < GPU[line][12+OFFSET]:
            NEWTON[line][15+OFFSET] = 0
        else:
            NEWTON[line][15+OFFSET] = 10
        if args.n_channel == 32 and NEWTON[line][13+OFFSET] == 0:
            NEWTON[line][13+OFFSET] = BASE[line][12+OFFSET]
        NEWTON[line][16+OFFSET] = BASE[line][12+OFFSET] / min(NEWTON[line][13+OFFSET], GPU[line][12+OFFSET])
        NEWTON[line][17+OFFSET] = MAX[line][16+OFFSET]

    with open(f'max_performance_{model}_{args.n_channel}_{args.n_gwrite}{postfix}.csv', 'w',newline='') as f:
        write = csv.writer(f)
        write.writerow(head)
        write.writerows(MAX)

    with open(f'newton_performance_{model}_{args.n_channel}_{args.n_gwrite}{postfix}.csv', 'w',newline='') as f:
        write = csv.writer(f)
        write.writerow(head + ["SPEED_UP (SPLIT)"])
        write.writerows(NEWTON)

process(args.model)
