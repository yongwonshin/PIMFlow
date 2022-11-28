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
N_CHANNEL=${14}
N_GWRITE=${15}

../pim/pim_codegen -oc $OUT_CHANNELS -ic $IN_CHANNELS -h $IMAGE_HEIGHT -w $IMAGE_WIDTH -kh $KH -kw $KW -ph $PH -pw $PW -stride $STRIDE -name PIM_trace_partition_$N_CHANNEL -n_channel $N_CHANNEL -gw $N_GWRITE

# rm -rf $NAME
mkdir -p $NAME

for i in ./PIM_trace_partition_$N_CHANNEL-*.pim ; do
    mv $i $NAME
done

