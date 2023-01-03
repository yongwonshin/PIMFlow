#!/bin/bash

NET=bert-large-1x64
echo "START!"
./pimflow -m=profile -t=split -n=$NET >split-$NET.log 2>split-$NET.err
# ./pimflow -m=stat --matmul_only -n=$NET
echo "END!"

NET=bert-large-1x32
echo "START!"
./pimflow -m=profile -t=split -n=$NET >split-$NET.log 2>split-$NET.err
# ./pimflow -m=stat --matmul_only -n=$NET
echo "END!"

NET=bert-large-1x3
echo "START!"
./pimflow -m=profile -t=split -n=$NET >split-$NET.log 2>split-$NET.err
# ./pimflow -m=stat --matmul_only -n=$NET
echo "END!"
