#!/bin/bash
MODEL=$1
GPU=$2
KERNEL_LAUNCH_LATENCY=$3
N_CHANNEL=$4

for (( i = 1; i < 4; i = i + 1)) ; do
  # python3 extract_layers.py --model=$MODEL --n_channel=$N_CHANNEL
  # python3 run --trace --gpgpusim_config=$GPU --model=$MODEL --pipeline=$i --kernel_launch_latency=$KERNEL_LAUNCH_LATENCY --n_channel=$N_CHANNEL
  python3 run --simulate --gpgpusim_config=$GPU --model=$MODEL --pipeline=$i --kernel_launch_latency=$KERNEL_LAUNCH_LATENCY --n_channel=$N_CHANNEL
  python3 run --stat --gpgpusim_config=$GPU --model=$MODEL --pipeline=$i --kernel_launch_latency=$KERNEL_LAUNCH_LATENCY --n_channel=$N_CHANNEL
  sh clean.sh $MODEL $i $N_CHANNEL
done
