FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04
ENV     DEBIAN_FRONTEND=noninteractive
ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
                    python3-dev \
                    ca-certificates \
                    g++ \
                    python3-numpy \
                    gcc \
                    make \
                    git \
                    python3-setuptools \
                    python3-wheel \
                    python3-pip \
                    aria2 \
                    wget \
                    build-essential \
                    xutils-dev \
                    bison \
                    zlib1g-dev \
                    flex \
                    libglu1-mesa-dev \
                    git \
                    libssl-dev \
                    libxml2-dev \
                    libboost-all-dev \
                    vim \
                    python-setuptools \
                    python-dev \
                    ninja-build \
                    bc \
                    git-lfs \
                    libtinfo-dev \
                    htop \
                    libedit-dev && \
    python3 -m pip install -U --force-reinstall pip && \
    #pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113 \
    pip3 install torch==1.12.1+cu113 \
                 torchvision==0.13.1+cu113 \
                 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113 \
                 pyyaml==5.1 \
                 onnx \
                 plotly \
                 psutil \
                 pandas \
                 decorator \
                 attrs \
                 scipy && \
    aria2c -q -d /tmp -o cmake-3.21.0-linux-x86_64.tar.gz \
           https://github.com/Kitware/CMake/releases/download/v3.21.0/cmake-3.21.0-linux-x86_64.tar.gz && \
    tar -zxf /tmp/cmake-3.21.0-linux-x86_64.tar.gz --strip=1 -C /usr && \
    wget -c https://github.com/llvm/llvm-project/releases/download/llvmorg-13.0.0/clang+llvm-13.0.0-x86_64-linux-gnu-ubuntu-20.04.tar.xz && \
    tar -xvf clang+llvm-13.0.0-x86_64-linux-gnu-ubuntu-20.04.tar.xz && \
    cp -rl clang+llvm-13.0.0-x86_64-linux-gnu-ubuntu-20.04/* /usr/local && \
    rm -rf clang+llvm-13.0.0-x86_64-linux-gnu-ubuntu-20.04 \
           clang+llvm-13.0.0-x86_64-linux-gnu-ubuntu-20.04.tar.xz
ENV HOME /root
WORKDIR /root
ENV TVM_HOME /root/PIMFlow_tvm
ENV PYTHONPATH /root/PIMFlow_tvm/python
ENV CUDA_INSTALL_PATH /usr/local/cuda
COPY install.sh /root/
CMD ["/bin/bash"]
