# Define base image.
FROM mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.3-cudnn8-ubuntu20.04:20240304.v1

# Variables used at build time.
## CUDA architectures, required by Colmap and tiny-cuda-nn.
## NOTE: All commonly used GPU architectures are included and supported here. To speedup the image build process remove all architectures but the one of your explicit GPU. Find details here: https://developer.nvidia.com/cuda-gpus (8.6 translates to 86 in the line below) or in the docs.
# ARG CUDA_ARCHITECTURES=90;89;86;80;75;70;61;52;37
ARG CUDA_ARCHITECTURES=75

# Set environment variables.
## Set non-interactive to prevent asking for user inputs blocking image creation.
ENV DEBIAN_FRONTEND=noninteractive
## Set timezone as it is required by some packages.
ENV TZ=Europe/Berlin
## CUDA Home, required to find CUDA in some packages.
ENV CUDA_HOME="/usr/local/cuda"
ENV CUB_HOME="/usr/local/cub"


ENV ACCEPT_EULA=Y

RUN apt --fix-missing update

RUN apt-get install software-properties-common -y

RUN add-apt-repository ppa:deadsnakes/ppa

RUN apt-get update -y

RUN apt-get install -y python3.10 -y

# Install required apt packages and clear cache afterwards.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    curl \
    ffmpeg \
    git \
    libatlas-base-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-program-options-dev \
    libboost-system-dev \
    libboost-test-dev \
    libhdf5-dev \
    libcgal-dev \
    libeigen3-dev \
    libflann-dev \
    libfreeimage-dev \
    libgflags-dev \
    libglew-dev \
    libgoogle-glog-dev \
    libmetis-dev \
    libprotobuf-dev \
    libqt5opengl5-dev \
    libsqlite3-dev \
    libsuitesparse-dev \
    nano \
    protobuf-compiler \
    python-is-python3 \
    python3.10 \
    python3-pip \
    qtbase5-dev \
    sudo \
    vim-tiny \
    wget && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install libopenblas-dev -y
RUN python3.10 -m pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

# Created with pip freeze -l > requirements.txt
COPY requirements.txt requirements.txt
#RUN conda install conda-forge::cudatoolkit
RUN python3.10 -m pip install -r requirements.txt

COPY plenoxels/ svox2
RUN cd svox2 &&     TORCH_CUDA_ARCH_LIST="7.5+PTX" python3.10 -m pip install -e . && \
    cd ..

# Create non root user and setup environment.
RUN useradd -m -d /home/user -g root -G sudo -u 1000 user
RUN usermod -aG sudo user
RUN usermod -aG video user
# Set user password
RUN echo "user:user" | chpasswd
# Ensure sudo group users are not asked for a password when using sudo command by ammending sudoers file
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

# Switch to new user and workdir.
USER 1000
WORKDIR /home/user

# Add local user binary folder to PATH variable.
ENV PATH="${PATH}:/home/user/.local/bin"
SHELL ["/bin/bash", "-c"]

RUN python3.10 -m pip freeze

# Change working directory
WORKDIR /workspace
