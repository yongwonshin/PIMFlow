#!/bin/bash
MODEL=$1
GPU=$2
KERNEL_LAUNCH_LATENCY=$3
N_CHANNEL=$4

for (( i = 1; i < 2; i = i + 1)) ; do
  # python extract_layers.py --model=$MODEL --n_channel=$N_CHANNEL
  # python run --trace --gpgpusim_config=$GPU --model=$MODEL --pipeline=$i --kernel_launch_latency=$KERNEL_LAUNCH_LATENCY --n_channel=$N_CHANNEL
  python run --simulate --gpgpusim_config=$GPU --model=$MODEL --pipeline=$i --kernel_launch_latency=$KERNEL_LAUNCH_LATENCY --n_channel=$N_CHANNEL
  python run --stat --gpgpusim_config=$GPU --model=$MODEL --pipeline=$i --kernel_launch_latency=$KERNEL_LAUNCH_LATENCY --n_channel=$N_CHANNEL
  sh clean.sh $MODEL $i $N_CHANNEL
done
