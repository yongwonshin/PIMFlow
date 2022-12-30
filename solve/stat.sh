#!/bin/bash
MODEL=$1
N_CHANNEL=$2

python3 stat.py --model=$MODEL --pipeline=all --n_channel=$N_CHANNEL --n_gwrite=1
python3 stat.py --model=$MODEL --pipeline=all --n_channel=$N_CHANNEL --n_gwrite=1 --ramulator_disable_gwrite_latency_hiding
python3 stat.py --model=$MODEL --pipeline=all --n_channel=$N_CHANNEL --n_gwrite=2
python3 stat.py --model=$MODEL --pipeline=all --n_channel=$N_CHANNEL --n_gwrite=2 --ramulator_disable_gwrite_latency_hiding
python3 stat.py --model=$MODEL --pipeline=all --n_channel=$N_CHANNEL --n_gwrite=4
python3 stat.py --model=$MODEL --pipeline=all --n_channel=$N_CHANNEL --n_gwrite=4 --ramulator_disable_gwrite_latency_hiding
