#!/bin/bash
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
