#!/bin/bash

NET=efficientnet-v1-b0
echo "START!"
./pimflow -m=profile -t=split -n=$NET >split-$NET.log 2>split-$NET.err
./pimflow -m=profile -t=pipeline -n=$NET >pipeline-$NET.log 2>pipeline-$NET.err
./pimflow -m=stat --conv_only -n=$NET
./pimflow -m=solve -n=$NET >solve-$NET.log 2>solve-$NET.err
./pimflow -m=run --gpu_only --policy=None -n=$NET >run_gpu_only-$NET.log 2>run_gpu_only-$NET.err
./pimflow -m=run --policy=Newton+ -n=$NET >run-$NET-Newton+.log 2>run-$NET-Newton+.err
./pimflow -m=run --policy=Newton++ -n=$NET >run-$NET-Newton++.log 2>run-$NET-Newton++.err
./pimflow -m=run --policy=MDDP -n=$NET >run-$NET-MDDP.log 2>run-$NET-MDDP.err
./pimflow -m=run --policy=Pipeline -n=$NET >run-$NET-Pipeline.log 2>run-$NET-Pipeline.err
./pimflow -m=run --policy=PIMFlow -n=$NET >run-$NET-PIMFlow.log 2>run-$NET-PIMFlow.err
./pimflow -m=stat -n=$NET
echo "END!"

NET=mobilenet-v2
echo "START!"
./pimflow -m=profile -t=split -n=$NET >split-$NET.log 2>split-$NET.err
./pimflow -m=profile -t=pipeline -n=$NET >pipeline-$NET.log 2>pipeline-$NET.err
./pimflow -m=stat --conv_only -n=$NET
./pimflow -m=solve -n=$NET >solve-$NET.log 2>solve-$NET.err
./pimflow -m=run --gpu_only --policy=None -n=$NET >run_gpu_only-$NET.log 2>run_gpu_only-$NET.err
./pimflow -m=run --policy=Newton+ -n=$NET >run-$NET-Newton+.log 2>run-$NET-Newton+.err
./pimflow -m=run --policy=Newton++ -n=$NET >run-$NET-Newton++.log 2>run-$NET-Newton++.err
./pimflow -m=run --policy=MDDP -n=$NET >run-$NET-MDDP.log 2>run-$NET-MDDP.err
./pimflow -m=run --policy=Pipeline -n=$NET >run-$NET-Pipeline.log 2>run-$NET-Pipeline.err
./pimflow -m=run --policy=PIMFlow -n=$NET >run-$NET-PIMFlow.log 2>run-$NET-PIMFlow.err
./pimflow -m=stat -n=$NET
echo "END!"

NET=mnasnet-1.0
echo "START!"
./pimflow -m=profile -t=split -n=$NET >split-$NET.log 2>split-$NET.err
./pimflow -m=profile -t=pipeline -n=$NET >pipeline-$NET.log 2>pipeline-$NET.err
./pimflow -m=stat --conv_only -n=$NET
./pimflow -m=solve -n=$NET >solve-$NET.log 2>solve-$NET.err
./pimflow -m=run --gpu_only --policy=None -n=$NET >run_gpu_only-$NET.log 2>run_gpu_only-$NET.err
./pimflow -m=run --policy=Newton+ -n=$NET >run-$NET-Newton+.log 2>run-$NET-Newton+.err
./pimflow -m=run --policy=Newton++ -n=$NET >run-$NET-Newton++.log 2>run-$NET-Newton++.err
./pimflow -m=run --policy=MDDP -n=$NET >run-$NET-MDDP.log 2>run-$NET-MDDP.err
./pimflow -m=run --policy=Pipeline -n=$NET >run-$NET-Pipeline.log 2>run-$NET-Pipeline.err
./pimflow -m=run --policy=PIMFlow -n=$NET >run-$NET-PIMFlow.log 2>run-$NET-PIMFlow.err
./pimflow -m=stat -n=$NET
echo "END!"

NET=resnet-50
echo "START!"
./pimflow -m=profile -t=split -n=$NET >split-$NET.log 2>split-$NET.err
./pimflow -m=profile -t=pipeline -n=$NET >pipeline-$NET.log 2>pipeline-$NET.err
./pimflow -m=stat --conv_only -n=$NET
./pimflow -m=solve -n=$NET >solve-$NET.log 2>solve-$NET.err
./pimflow -m=run --gpu_only --policy=None -n=$NET >run_gpu_only-$NET.log 2>run_gpu_only-$NET.err
./pimflow -m=run --policy=Newton+ -n=$NET >run-$NET-Newton+.log 2>run-$NET-Newton+.err
./pimflow -m=run --policy=Newton++ -n=$NET >run-$NET-Newton++.log 2>run-$NET-Newton++.err
./pimflow -m=run --policy=MDDP -n=$NET >run-$NET-MDDP.log 2>run-$NET-MDDP.err
./pimflow -m=run --policy=Pipeline -n=$NET >run-$NET-Pipeline.log 2>run-$NET-Pipeline.err
./pimflow -m=run --policy=PIMFlow -n=$NET >run-$NET-PIMFlow.log 2>run-$NET-PIMFlow.err
./pimflow -m=stat -n=$NET
echo "END!"

NET=vgg-16
echo "START!"
./pimflow -m=profile -t=split -n=$NET >split-$NET.log 2>split-$NET.err
./pimflow -m=profile -t=pipeline -n=$NET >pipeline-$NET.log 2>pipeline-$NET.err
./pimflow -m=stat --conv_only -n=$NET
./pimflow -m=solve -n=$NET >solve-$NET.log 2>solve-$NET.err
./pimflow -m=run --gpu_only --policy=None -n=$NET >run_gpu_only-$NET.log 2>run_gpu_only-$NET.err
./pimflow -m=run --policy=Newton+ -n=$NET >run-$NET-Newton+.log 2>run-$NET-Newton+.err
./pimflow -m=run --policy=Newton++ -n=$NET >run-$NET-Newton++.log 2>run-$NET-Newton++.err
./pimflow -m=run --policy=MDDP -n=$NET >run-$NET-MDDP.log 2>run-$NET-MDDP.err
./pimflow -m=run --policy=Pipeline -n=$NET >run-$NET-Pipeline.log 2>run-$NET-Pipeline.err
./pimflow -m=run --policy=PIMFlow -n=$NET >run-$NET-PIMFlow.log 2>run-$NET-PIMFlow.err
./pimflow -m=stat -n=$NET
echo "END!"
