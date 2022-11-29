#!/bin/bash
MODEL=$1
GPU=$2
KERNEL_LAUNCH_LATENCY=$3
N_CHANNEL=$4

for (( i = 10 ; i < 101 ; i = i + 10 )) ; do
    if [ $i -eq 50 ] || [ $i -eq 100 ]
    then
	    python3 inspect_shape.py --model=$MODEL --split_ratio=$i --n_channel=$N_CHANNEL
	    python3 run --trace --gpgpusim_config=$GPU --model=$MODEL --split_ratio=$i --kernel_launch_latency=$KERNEL_LAUNCH_LATENCY --n_channel=$N_CHANNEL
	    python3 run --simulate --gpgpusim_config=$GPU --model=$MODEL --split_ratio=$i --kernel_launch_latency=$KERNEL_LAUNCH_LATENCY --n_channel=$N_CHANNEL
	    python3 run --stat --gpgpusim_config=$GPU --model=$MODEL --split_ratio=$i --kernel_launch_latency=$KERNEL_LAUNCH_LATENCY --n_channel=$N_CHANNEL
    fi
    if [ $i -eq 100 ]
    then
        python3 run --pim --gpgpusim_config=$GPU --model=$MODEL --n_channel=$N_CHANNEL
    fi
    sh clean.sh $MODEL $i $N_CHANNEL
done
