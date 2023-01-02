#!/bin/bash

NAME=$1
ROW=$2
COL=$3
BIAS=$4
ACTIVATION=$5
BATCH_SIZE=$6
GPU=$7
RATIO=$8
N_CHANNEL=$9

EXTRA_GPU_CONFIG_1="-gpgpu_n_mem $((32-$N_CHANNEL)) -gpgpu_deadlock_detect 0"
EXTRA_GPU_CONFIG_2="-gpgpu_n_mem 32 -gpgpu_deadlock_detect 0"

BASE_PATH="/root/PIMFlow_accel-sim-framework"

export CUDA_INSTALL_PATH=/usr/local/cuda
source "$BASE_PATH/gpu-simulator/setup_environment.sh"
if [ $RATIO -eq 100 ]
then
    timeout 21600 $BASE_PATH/gpu-simulator/bin/release/accel-sim.out -trace "traces-matmul-$NAME/kernelslist.g" -config "$BASE_PATH/gpu-simulator/configs/tested-cfgs/$GPU/trace.config" -config "$BASE_PATH/gpu-simulator/gpgpu-sim/configs/tested-cfgs/$GPU/gpgpusim.config" $EXTRA_GPU_CONFIG_2 | grep -E "kernel_name|gpu_sim_cycle|gpu_tot_sim_cycle" &> traces-matmul-$NAME-baseline.txt
    timeout 21600 $BASE_PATH/gpu-simulator/bin/release/accel-sim.out -trace "traces-matmul-$NAME/kernelslist.g" -config "$BASE_PATH/gpu-simulator/configs/tested-cfgs/$GPU/trace.config" -config "$BASE_PATH/gpu-simulator/gpgpu-sim/configs/tested-cfgs/$GPU/gpgpusim.config" $EXTRA_GPU_CONFIG_1 | grep -E "kernel_name|gpu_sim_cycle|gpu_tot_sim_cycle" &> traces-matmul-$NAME.txt
else
    timeout 21600 $BASE_PATH/gpu-simulator/bin/release/accel-sim.out -trace "traces-matmul-$NAME/kernelslist.g" -config "$BASE_PATH/gpu-simulator/configs/tested-cfgs/$GPU/trace.config" -config "$BASE_PATH/gpu-simulator/gpgpu-sim/configs/tested-cfgs/$GPU/gpgpusim.config" $EXTRA_GPU_CONFIG_1 | grep -E "kernel_name|gpu_sim_cycle|gpu_tot_sim_cycle" &> traces-matmul-$NAME.txt
fi
