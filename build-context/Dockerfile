# Define base image.
FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04

# Variables used at build time.
## CUDA architectures, required by Colmap and tiny-cuda-nn.
ARG CUDA_ARCHITECTURES=75

# Set environment variables.
## Set non-interactive to prevent asking for user inputs blocking image creation.
ENV DEBIAN_FRONTEND=noninteractive
## Set timezone as it is required by some packages.
ENV TZ=Europe/Berlin
## CUDA Home, required to find CUDA in some packages.
ENV CUDA_HOME="/usr/local/cuda"
## Point PETSc to the installed compilers
ENV PATH="/usr/bin:$PATH"
ENV CC=/usr/bin/mpicc CXX=/usr/bin/mpicxx FC=/usr/bin/mpifort

# Install required apt packages and clear cache afterwards.
RUN apt update -y
RUN apt install build-essential -y
RUN apt install cmake -y
RUN apt install make -y
RUN apt install git -y
RUN apt install python3-dev -y
RUN apt install sudo -y
RUN apt install gfortran -y
RUN apt install mpich -y
RUN apt install libmpich-dev -y
RUN apt install wget -y
RUN apt install libblas-dev liblapack-dev petsc-dev -y
RUN apt install -y \
    libgl1 \
    libglu1-mesa \
    libxrandr2 \
    libxinerama1 \
    libxcursor1 \
    libxft-dev \
    libfreetype6 \
    libfontconfig1
RUN apt install python3-venv -y
RUN rm -rf /var/lib/apt/lists/*

RUN apt update -y
RUN apt install -y --no-install-recommends curl ca-certificates
RUN apt install -y python3-pip
RUN pip install uv --break-system-packages
RUN uv --version

# Create non root user and setup environment.
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

# Switch to new user and workdir.
USER root
WORKDIR /home/ubuntu

# Add local user binary folder to PATH variable.
ENV PATH="${PATH}:/home/ubuntu/.local/bin"
SHELL ["/bin/bash", "-c"]


COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml
COPY .python-version .python-version
COPY README.md README.md
RUN ls

# Install dependencies

RUN export PEP517_BUILD_BACKEND=setuptools.build_meta && uv sync --frozen --extra azure --extra fem

RUN uv pip freeze
# Put the python binary at the front of the path to use directly in future commands.
ENV PATH="/home/ubuntu/.venv/bin:$PATH"
# Change working directory
WORKDIR /workspace

