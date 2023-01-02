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

export DYNAMIC_KERNEL_LIMIT_START=1000000000
LD_PRELOAD=/root/PIMFlow_accel-sim-framework/util/tracer_nvbit/tracer_tool/tracer_tool.so python3 /root/PIMFlow/layerwise/layerwise_matmul.py --name=$NAME --batch=$BATCH_SIZE --row=$ROW --col=$COL --bias=$BIAS --activation=$ACTIVATION

START=$(python3 inspect --path=traces-matmul-$NAME | sed -r 's/([0-9]*)\s([0-9]*)/\1/g')
END=$(python3 inspect --path=traces-matmul-$NAME | sed -r 's/([0-9]*)\s([0-9]*)/\2/g')
export DYNAMIC_KERNEL_LIMIT_START=$START
export DYNAMIC_KERNEL_LIMIT_END=$END

LD_PRELOAD=/root/PIMFlow_accel-sim-framework/util/tracer_nvbit/tracer_tool/tracer_tool.so python3 /root/PIMFlow/layerwise/layerwise_matmul.py --name=$NAME --batch=$BATCH_SIZE --row=$ROW --col=$COL --bias=$BIAS --activation=$ACTIVATION

/root/PIMFlow_accel-sim-framework/util/tracer_nvbit/tracer_tool/traces-processing/post-traces-processing ./traces-matmul-$NAME/kernelslist
