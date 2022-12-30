#!/bin/bash
MODEL=$1
GPU=$2
KERNEL_LAUNCH_LATENCY=$3
N_CHANNEL=$4

for (( i = 1; i < 4; i = i + 1)) ; do
  python3 extract_layers.py --model=$MODEL --n_channel=$N_CHANNEL
  python3 run --trace --gpgpusim_config=$GPU --model=$MODEL --pipeline=$i --kernel_launch_latency=$KERNEL_LAUNCH_LATENCY --n_channel=$N_CHANNEL
  python3 run --simulate --gpgpusim_config=$GPU --model=$MODEL --pipeline=$i --kernel_launch_latency=$KERNEL_LAUNCH_LATENCY --n_channel=$N_CHANNEL
  python3 run --pim_codegen --model=$MODEL --pipeline=$i --n_channel=$N_CHANNEL --n_gwrite=1
  python3 run --pim_codegen --model=$MODEL --pipeline=$i --n_channel=$N_CHANNEL --n_gwrite=2
  python3 run --pim_codegen --model=$MODEL --pipeline=$i --n_channel=$N_CHANNEL --n_gwrite=4
  python3 run --stat --gpgpusim_config=$GPU --model=$MODEL --pipeline=$i --kernel_launch_latency=$KERNEL_LAUNCH_LATENCY --n_channel=$N_CHANNEL --n_gwrite=1
  python3 run --stat --gpgpusim_config=$GPU --model=$MODEL --pipeline=$i --kernel_launch_latency=$KERNEL_LAUNCH_LATENCY --n_channel=$N_CHANNEL --n_gwrite=2
  python3 run --stat --gpgpusim_config=$GPU --model=$MODEL --pipeline=$i --kernel_launch_latency=$KERNEL_LAUNCH_LATENCY --n_channel=$N_CHANNEL --n_gwrite=4
  sh clean.sh $MODEL $i $N_CHANNEL
done

python3 to_full_layer.py --model $MODEL --n_gwrite=1
python3 to_full_layer.py --model $MODEL --n_gwrite=1 --ramulator_disable_gwrite_latency_hiding
python3 to_full_layer.py --model $MODEL --n_gwrite=2
python3 to_full_layer.py --model $MODEL --n_gwrite=2 --ramulator_disable_gwrite_latency_hiding
python3 to_full_layer.py --model $MODEL --n_gwrite=4
python3 to_full_layer.py --model $MODEL --n_gwrite=4 --ramulator_disable_gwrite_latency_hiding
