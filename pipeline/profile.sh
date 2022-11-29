#!/bin/bash
MODEL=$1

bash ALL.sh $MODEL SM75_RTX2060 5010 16
python3 to_full_layer.py --model $MODEL
