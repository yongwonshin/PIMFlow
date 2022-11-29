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
ACTIVATION=${15}
DEVICE_ID=${16}
N_CHANNEL=${17}

# assume docker is setup
export TVM_HOME=/root/tvm
export PYTHONPATH=$TVM_HOME/python:${PYTHONPATH}

export DYNAMIC_KERNEL_LIMIT_START=1000000000
LD_PRELOAD=/root/PIMFlow_accel-sim-framework/util/tracer_nvbit/tracer_tool/tracer_tool.so python3 /root/PIMFlow/pipeline/layerwise.py --oc=$OUT_CHANNELS --ic=$IN_CHANNELS --kh=$KH --kw=$KW --stride=$STRIDE --ph=$PH --pw=$PW --dilate=$DILATE --g=$GROUP --b --h=$IMAGE_HEIGHT --w=$IMAGE_WIDTH --dev=$CUDA_VISIBLE_DEVICES --activation=$ACTIVATION

START=$(python3 inspect --path=traces-$NAME | sed -r 's/([0-9]*)\s([0-9]*)/\1/g')
END=$(python3 inspect --path=traces-$NAME | sed -r 's/([0-9]*)\s([0-9]*)/\2/g')
export DYNAMIC_KERNEL_LIMIT_START=$START
export DYNAMIC_KERNEL_LIMIT_END=$END

LD_PRELOAD=/root/PIMFlow_accel-sim-framework/util/tracer_nvbit/tracer_tool/tracer_tool.so python3 /root/PIMFlow/pipeline/layerwise.py --oc=$OUT_CHANNELS --ic=$IN_CHANNELS --kh=$KH --kw=$KW --stride=$STRIDE --ph=$PH --pw=$PW --dilate=$DILATE --g=$GROUP --b --h=$IMAGE_HEIGHT --w=$IMAGE_WIDTH --dev=$CUDA_VISIBLE_DEVICES --activation=$ACTIVATION

/root/PIMFlow_accel-sim-framework/util/tracer_nvbit/tracer_tool/traces-processing/post-traces-processing ./traces-$NAME/kernelslist
