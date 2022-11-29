#!/bin/bash

NAME=$1
OUT_CHANNELS=$2
IN_CHANNELS=$3
KH=$4
KW=$5
STRIDE=$6
PH=$7
PW=$8
DILATE=$9
GROUP=${10}
BIAS=${11}
IMAGE_HEIGHT=${12}
IMAGE_WIDTH=${13}
GPU=${14}
RATIO=${15}
ACTIVATION=${16}
DEVICE_ID=${17}
N_CHANNEL=${18}

EXTRA_GPU_CONFIG_1="-gpgpu_n_mem $((32-$N_CHANNEL)) -gpgpu_deadlock_detect 0"
EXTRA_GPU_CONFIG_2="-gpgpu_n_mem 32 -gpgpu_deadlock_detect 0"
# TODO: add PIM_PATH

BASE_PATH="/root/PIMFlow_accel-sim-framework"

export CUDA_INSTALL_PATH=/usr/local/cuda
source "$BASE_PATH/gpu-simulator/setup_environment.sh"
if [ $RATIO -eq 100 ]
then
    timeout 21600 $BASE_PATH/gpu-simulator/bin/release/accel-sim.out -trace "traces-$NAME/kernelslist.g" -config "$BASE_PATH/gpu-simulator/configs/tested-cfgs/$GPU/trace.config" -config "$BASE_PATH/gpu-simulator/gpgpu-sim/configs/tested-cfgs/$GPU/gpgpusim.config" $EXTRA_GPU_CONFIG_2 | grep -E "kernel_name|gpu_sim_cycle|gpu_tot_sim_cycle" &> traces-$NAME-baseline.txt
    timeout 21600 $BASE_PATH/gpu-simulator/bin/release/accel-sim.out -trace "traces-$NAME/kernelslist.g" -config "$BASE_PATH/gpu-simulator/configs/tested-cfgs/$GPU/trace.config" -config "$BASE_PATH/gpu-simulator/gpgpu-sim/configs/tested-cfgs/$GPU/gpgpusim.config" $EXTRA_GPU_CONFIG_1 | grep -E "kernel_name|gpu_sim_cycle|gpu_tot_sim_cycle" &> traces-$NAME.txt
else
    timeout 21600 $BASE_PATH/gpu-simulator/bin/release/accel-sim.out -trace "traces-$NAME/kernelslist.g" -config "$BASE_PATH/gpu-simulator/configs/tested-cfgs/$GPU/trace.config" -config "$BASE_PATH/gpu-simulator/gpgpu-sim/configs/tested-cfgs/$GPU/gpgpusim.config" $EXTRA_GPU_CONFIG_1 | grep -E "kernel_name|gpu_sim_cycle|gpu_tot_sim_cycle" &> traces-$NAME.txt
fi
