#!/bin/bash
MODEL=$1
GPU=$2
KERNEL_LAUNCH_LATENCY=$3
N_CHANNEL=$4

for (( i = 10 ; i < 101 ; i = i + 10 )) ; do
    python inspect_shape.py --model=$MODEL --split_ratio=$i --n_channel=$N_CHANNEL
    python run --trace --gpgpusim_config=$GPU --model=$MODEL --split_ratio=$i --kernel_launch_latency=$KERNEL_LAUNCH_LATENCY --n_channel=$N_CHANNEL
    python run --simulate --gpgpusim_config=$GPU --model=$MODEL --split_ratio=$i --kernel_launch_latency=$KERNEL_LAUNCH_LATENCY --n_channel=$N_CHANNEL
    python run --stat --gpgpusim_config=$GPU --model=$MODEL --split_ratio=$i --kernel_launch_latency=$KERNEL_LAUNCH_LATENCY --n_channel=$N_CHANNEL
    if [ $i -eq 100 ]
    then
        python run --pim --gpgpusim_config=$GPU --model=$MODEL --n_channel=$N_CHANNEL
    fi
    sh clean.sh $MODEL $i $N_CHANNEL
done
