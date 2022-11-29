#!/bin/bash
MODEL=$1
SPLIT=$2
N_CHANNEL=$3

mkdir -p result_simulate/$MODEL/${SPLIT}_${N_CHANNEL}
mv trace*.txt result_simulate/$MODEL/${SPLIT}_${N_CHANNEL}
mv *-matmul result_simulate/$MODEL/${SPLIT}_${N_CHANNEL}
mv Conv_* result_simulate/$MODEL/${SPLIT}_${N_CHANNEL}
mv Gemm_* result_simulate/$MODEL/${SPLIT}_${N_CHANNEL}
mv accelwattch_power_report_*.log result_simulate/$MODEL/${SPLIT}_${N_CHANNEL}
mv traces-* result_simulate/$MODEL/${SPLIT}_${N_CHANNEL}
rm -r traces-*
rm -r Conv_*
rm -r tmp-*
rm -r Gemm_*
rm compile-*.so
rm layer-*.onnx
rm MatMul_*.onnx
