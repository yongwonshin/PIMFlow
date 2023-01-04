#!/bin/bash
MODEL=$1
N_CHANNEL=$2
STAGE=$3

python3 stat.py --model=$MODEL --pipeline=all --n_channel=$N_CHANNEL --n_gwrite=1 --stage=$STAGE
python3 stat.py --model=$MODEL --pipeline=all --n_channel=$N_CHANNEL --n_gwrite=1 --ramulator_disable_gwrite_latency_hiding --stage=$STAGE
python3 stat.py --model=$MODEL --pipeline=all --n_channel=$N_CHANNEL --n_gwrite=2 --stage=$STAGE
python3 stat.py --model=$MODEL --pipeline=all --n_channel=$N_CHANNEL --n_gwrite=2 --ramulator_disable_gwrite_latency_hiding --stage=$STAGE
python3 stat.py --model=$MODEL --pipeline=all --n_channel=$N_CHANNEL --n_gwrite=4 --stage=$STAGE
python3 stat.py --model=$MODEL --pipeline=all --n_channel=$N_CHANNEL --n_gwrite=4 --ramulator_disable_gwrite_latency_hiding --stage=$STAGE
