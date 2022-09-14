#!/bin/bash
#SBATCH -J matsci-train
#SBATCH -N 2
#SBATCH --jobs-per-node 4
#SBATCH --bind-py numa
#SBATCH -t 12:00:00

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: MIT License

# this batch script is used to launch a CPU-based training job
# on a Slurm cluster

# bind-by=numa will localize CPU resource utilization
# to NUMA domains, minimizing non-uniform memory access

# configure thread affinity and number of threads; depends
# on the CPU available
export OMP_NUM_THREADS=36
export KMP_BLOCKTIME=5

# launch the training script
python simple_example_slurm.py
