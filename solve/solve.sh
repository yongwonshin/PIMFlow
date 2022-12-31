#!/bin/bash
MODEL=$1
N_CHANNEL=$2


python3 solve.py --model=$MODEL --pipeline=all --n_channel=$N_CHANNEL --n_gwrite=1 --ramulator_disable_gwrite_latency_hiding --policy=Newton+
python3 solve.py --model=$MODEL --pipeline=all --n_channel=$N_CHANNEL --n_gwrite=2 --ramulator_disable_gwrite_latency_hiding --policy=Newton+
python3 solve.py --model=$MODEL --pipeline=all --n_channel=$N_CHANNEL --n_gwrite=4 --ramulator_disable_gwrite_latency_hiding --policy=Newton+

python3 solve.py --model=$MODEL --pipeline=all --n_channel=$N_CHANNEL --n_gwrite=1 --policy=Newton++
python3 solve.py --model=$MODEL --pipeline=all --n_channel=$N_CHANNEL --n_gwrite=2 --policy=Newton++
python3 solve.py --model=$MODEL --pipeline=all --n_channel=$N_CHANNEL --n_gwrite=4 --policy=Newton++


python3 solve.py --model=$MODEL --pipeline=all --n_channel=$N_CHANNEL --n_gwrite=1 --policy=Pipeline
python3 solve.py --model=$MODEL --pipeline=all --n_channel=$N_CHANNEL --n_gwrite=2 --policy=Pipeline
python3 solve.py --model=$MODEL --pipeline=all --n_channel=$N_CHANNEL --n_gwrite=4 --policy=Pipeline

python3 solve.py --model=$MODEL --pipeline=all --n_channel=$N_CHANNEL --n_gwrite=1 --policy=MDDP
python3 solve.py --model=$MODEL --pipeline=all --n_channel=$N_CHANNEL --n_gwrite=2 --policy=MDDP
python3 solve.py --model=$MODEL --pipeline=all --n_channel=$N_CHANNEL --n_gwrite=4 --policy=MDDP

python3 solve.py --model=$MODEL --pipeline=all --n_channel=$N_CHANNEL --n_gwrite=1 --policy=PIMFlow
python3 solve.py --model=$MODEL --pipeline=all --n_channel=$N_CHANNEL --n_gwrite=2 --policy=PIMFlow
python3 solve.py --model=$MODEL --pipeline=all --n_channel=$N_CHANNEL --n_gwrite=4 --policy=PIMFlow
