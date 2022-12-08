# PIMFlow

## Pre-requisites
### Hardware dependencies
We've tested codes on NVIDIA GeForce RTX 2080 Ti GPU in Ubuntu 20.04 amd64 system. GPU should have Turing architecture.

Note: GPU architecture over Turing could have bug when tracing (NVBit bug).

### System dependencies
We tested our code on Ubuntu 20.04 amd64 system, and used CUDA 11.3.1 and cuDNN 8.

### Software dependencies
Software pre-requisites for installing from the source should be satisfied for the following repositories:
- [TVM](https://github.com/apache/tvm)
- [GPGPU-Sim](https://github.com/gpgpu-sim/gpgpu-sim_distribution)
- [Accel-Sim](https://github.com/accel-sim/accel-sim-framework)
- [Ramulator](https://github.com/CMU-SAFARI/ramulator)

You can install all dependencies by following this document.

Firstly, make sure CUDA is installed in your system:
```bash
export CUDA_INSTALL_PATH=/usr/local/cuda # set it to your CUDA installation path
nvcc --version
```

In Ubuntu 20.04 amd64 system, following commands install package dependencies:
```bash
sudo apt-get update
sudo apt-get install -y --no-install-recommends python3-dev ca-certificates g++ python3-numpy gcc make git python3-setuptools python3-wheel python3-pip aria2 wget build-essential xutils-dev bison zlib1g-dev flex libglu1-mesa-dev git libssl-dev libxml2-dev libboost-all-dev vim python-setuptools python-dev ninja-build bc git-lfs libtinfo-dev htop libedit-dev
```

Next, install Python (>= 3.8) dependencies.

Note: you need to use specific PyTorch version (= 1.12.1). Later version could generate different node name that cannot be processed by the current version.

```bash
python3 -m pip install -U --force-reinstall pip
pip install  torch==1.11.0+cu113 \
             torchvision==0.12.0+cu113 \
             torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip3 install pyyaml==5.1 onnx plotly psutil pandas decorator attrs scipy
```

Install CMake (>= 3.21):
```bash
sudo aria2c -q -d /tmp -o cmake-3.21.0-linux-x86_64.tar.gz \
           https://github.com/Kitware/CMake/releases/download/v3.21.0/cmake-3.21.0-linux-x86_64.tar.gz
sudo tar -zxf /tmp/cmake-3.21.0-linux-x86_64.tar.gz --strip=1 -C /usr
```

Install Clang and LLVM (>= 12)
```bash
wget -c https://github.com/llvm/llvm-project/releases/download/llvmorg-13.0.0/clang+llvm-13.0.0-x86_64-linux-gnu-ubuntu-20.04.tar.xz
tar -xvf clang+llvm-13.0.0-x86_64-linux-gnu-ubuntu-20.04.tar.xz
sudo cp -rl clang+llvm-13.0.0-x86_64-linux-gnu-ubuntu-20.04/* /usr/local
rm -rf clang+llvm-13.0.0-x86_64-linux-gnu-ubuntu-20.04 \
       clang+llvm-13.0.0-x86_64-linux-gnu-ubuntu-20.04.tar.xz
```

## Setup

Install and build PIMFlow repositories from the source. We prepared installation script (docker/install.sh):
```bash
GIT_BRANCH="2023cgo-artifact"
cd "$HOME"
git clone -b "$GIT_BRANCH" https://github.com/yongwonshin/PIMFlow_tvm.git
TVM_DIR="$HOME/PIMFlow_tvm"
cd "$TVM_DIR"
git submodule init && git submodule update
BUILD_DIR="$TVM_DIR/build"
mkdir -p "$BUILD_DIR" && cd "$BUILD_DIR"
cp "$TVM_DIR/cmake/config.cmake" "$BUILD_DIR"
cmake .. -G Ninja -DCMAKE_CXX_COMPILER=$(which g++) -DCMAKE_C_COMPILER=$(which gcc)
ninja
cd "$HOME"
git clone -b "$GIT_BRANCH" https://github.com/yongwonshin/PIMFlow_accel-sim-framework.git
GPU_DIR="$HOME/PIMFlow_accel-sim-framework/gpu-simulator"
NVBIT_DIR="$HOME/PIMFlow_accel-sim-framework/util/tracer_nvbit"
cd "$GPU_DIR"
source setup_environment.sh
# Generate binary file: $GPU_DIR/bin/release/accel-sim.out
make -j
# Install nvbit
cd "$NVBIT_DIR" && ./install_nvbit.sh && make -j

cd "$HOME"
git clone -b "$GIT_BRANCH" https://github.com/yongwonshin/PIMFlow_ramulator.git
RAM_DIR="$HOME/PIMFlow_ramulator"
cd "$RAM_DIR"
# Generate binary file: $RAM_DIR/ramulator
make -j
cd "$HOME"
git clone https://github.com/yongwonshin/PIMFlow.git
PIMFLOW_DIR="$HOME/PIMFlow"
cd "$PIMFLOW_DIR"
pip install -e .
cd "$PIMFLOW_DIR/pim"
# Generate binary file: $PIMFLOW_DIR/pim/pim_codegen
make -j
# Extract mobilenet trace
cd "$PIMFLOW_DIR"
tar -xzf ./data/mobilenet-v2.tar.gz -C .
```

Now, the directory should look like this:
```text
. ($HOME)
./PIMFlow
./PIMFlow_tvm
./PIMFlow_gpgpu_sim_distribution
./PIMFlow_accel-sim-framework
./PIMFlow_ramulator
```

Finally, you need to set the following environment variables, and include them to .bashrc for later session.
```bash
export TVM_HOME=/root/PIMFlow_tvm
export PYTHONPATH=/root/PIMFlow_tvm/python
```

## Run
You can manually peform profiling to find optimal execution mode and task size.

Note: it takes about 8 hours in server with 8x NVIDIA GeForce RTX 2080 Ti GPU and 2x Intel Xeon Gold 6248R CPU (24-core)

```bash
cd PIMFlow
./pimflow -m=profile -t=split -n=mobilenet-v2
./pimflow -m=profile -t=pipeline -n=mobilenet-v2
```
Or, you can just use the profiled data we've prepared in PIMFlow/mobilenet-v2/ for MobileNet-V2.

Now, you can get the optimal solution using profiled data and get the speedup:
```bash
./pimflow -m=solve -n=mobilenet-v2
./pimflow -m=stat --conv_only -n=mobilenet-v2
```
The output should look like this:
```text
newton++ (vs baseline): 1.39 (-412549.3529238198)
pipeline (vs baseline): 1.457 (-461786.4352791244)
split (vs baseline): 1.461 (-464203.9705751848)
all (vs baseline): 1.515 (-500322.43528344436)
```

Next, you can get speedup by the following commands:
Note: it takes about 8 hours in our system.
```bash
./pimflow -m=run --gpu_only -n=mobilenet-v2 # get gpu-only execution time
./pimflow -m=run -n=mobilenet-v2 # get pimflow execution time
./pimflow -m=stat -n=mobilenet-v2 # show end-to-end speedup
```
Output:
```text
GPU CYCLE: 4390335
PIMFLOW CYCLE: 3122474.405912899
PIMFLOW SPEEDUP: 1.41
```

You can replace "mobilenet-v2" with "efficientnet-v1-b0", "mnasnet-1.0", "resnet-50" or "vgg-16" for various network testing.
We prepared very simple network "toy" for simple but fast test.
