#!/bin/bash -l
#SBATCH --partition=pvc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH -t 02:30:00
#SBATCH --mem-bind=local
#SBATCH --overcommit

# This is a Slurm batch file that launches XPU DDP
# training via Slurm and MPI, pairing with `xpu_ddp_mpi.py`.
# Calling `mpirun` means that you are not required to use
# Slurm, but we use it here for the ease of obtaining environment
# variables and whatnot.

# clean slate, load runtime libraries
# if module files are not available, source oneAPI directly.
# usually, you can run `source /opt/intel/oneapi/2024.0.0/setvars.sh`
# to replace all of the module loads.
module purge
module load default-dawn
module load intel-oneapi-compilers
module load intelpython-conda
module load intel-oneapi-mkl
module load intel-oneapi-mpi
module load intel-oneapi-ccl

# activate conda environment for matsciml
conda activate matsciml

export ZE_FLAT_DEVICE_HIERARCHY=FLAT

# this ulimit is generally set because of how PyTorch
# parallel data loader workers are implemented
ulimit -n 60000

# produce a nodefile for MPI from Slurm
scontrol show hostnames >hostfile

# map environment variables to MPI
mpirun -n $SLURM_NTASKS \
	-ppn $SLURM_NTASKS_PER_NODE \
	-f hostfile \
	-bootstrap ssh \
	-bootstrap-exec-args "-t" \
	-map-by socket \
	python xpu_ddp_mpi.py
