# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: MIT License

# CUDA version from nvidia docker: https://hub.docker.com/r/nvidia/cuda/tags
ARG CUDA_VERSION=11.8.0-cudnn8-runtime-ubuntu20.04
FROM nvidia/cuda:$CUDA_VERSION

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
# Avoids some interactive prompts during apt-get install
ARG DEBIAN_FRONTEND=noninteractive

# clean up and refresh apt-get index
RUN apt-get update && \
    apt-get autoremove -y && \
    apt-get clean && rm -rf /var/cache/apt/archives /var/lib/apt/lists/*

# Install system tools
RUN apt-get update --fix-missing && \
    apt-get install -yq --no-install-recommends \
    sudo \
    curl \
    wget \
    build-essential \
    git \
    unzip \
    tmux \
    screen \
    openssh-client && \
    apt-get clean && rm -rf /var/cache/apt/archives /var/lib/apt/lists/*

RUN apt-get update --fix-missing && \
    apt-get install -yq --no-install-recommends \
    sudo \
    software-properties-common \
    python3.9 \
    python3-pip \
    virtualenv && \
    apt-get clean && rm -rf /var/cache/apt/archives /var/lib/apt/lists/*

RUN apt-get update \
    && apt-get install -y wget tmux vim libxrender-dev libsm6 libxext6 && \
    apt-get clean && rm -rf /var/cache/apt/archives /var/lib/apt/lists/*

# Create py3 virtualenv in order to use python3 for the project
RUN virtualenv -p python3.9 /venv
ENV PATH=/venv/bin:$PATH
RUN pip install --upgrade pip
RUN pip install -q \
    ipdb \
    pandas \
    matplotlib \
    scikit-learn

RUN pip install numba lmdb munch geometric_algebra_attention jsonargparse[signatures] sympy pymatgen
RUN pip install torch==2.0.0+cu118 --index-url https://download.pytorch.org/whl/cu118
RUN pip install dgl==0.9.1 -f https://data.dgl.ai/wheels/cu118/repo.html
RUN pip install  dglgo -f https://data.dgl.ai/wheels-test/repo.html
RUN pip install dgllife==0.3.2 rdkit==2023.3.1 rowan
RUN pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
RUN pip install torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
RUN pip install torch-geometric
RUN pip install smact
RUN pip install matminer
RUN pip install p_tqdm
RUN pip install -U pytorch-lightning==1.8.6
RUN pip install -U torchmetrics==0.11.4
RUN pip install -U pytest
