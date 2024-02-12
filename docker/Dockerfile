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

RUN pip install psutil
RUN pip install pytorch_lightning==2.1.3
RUN pip install dgllife==0.3.2
RUN pip install rdkit==2023.3.1
RUN pip install numba>=0.55.0
RUN pip install numpy>=1.21.0
RUN pip install matplotlib
RUN pip install munch==2.5.0
RUN pip install lmdb==1.4.1
RUN pip install geometric-algebra-attention>=0.3.0
RUN pip install sympy>=1.10.1
RUN pip install jsonargparse[signatures]>=4.13.1
RUN pip install --no-cache mp_api==0.33.3
RUN pip install emmet-core==0.64.0
RUN pip install pydantic==1.10.12
RUN pip install pymatgen==2023.7.20
RUN pip install schema>=0.7.5
RUN pip install ase>=3.22.1
RUN pip install matgl==0.9.2
RUN pip install einops==0.7.0
RUN pip install mendeleev==0.14.0
RUN pip install e3nn==0.5.1

RUN pip install black>=22.3.0
RUN pip install flake8
RUN pip install flake8-pyproject
RUN pip install flake8-black
RUN pip install flake8-bandit
RUN pip install pre-commit
RUN pip install bandit
RUN pip install pytest

RUN pip install torch-scatter -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
RUN pip install torch-sparse -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
RUN pip install torch-geometric

RUN pip uninstall -y dgl
RUN pip install dgl -f https://data.dgl.ai/wheels/cu121/repo.html
