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
# training via Slurm, pairing with the `xpu_ddp_slurm.py` script.
# To change the number of workers, use the Slurm
# arguments above (`nodes`, `ntasks-per-node`) and should allow
# you to seamlessly scale up and scale out.

# clean slate, load runtime libraries
module purge
module load default-dawn
module load intel-oneapi-compilers
module load intelpython-conda
module load intel-oneapi-mkl
module load intel-oneapi-mpi
module load intel-oneapi-ccl

# activate conda environment for matsciml
conda activate matsciml

export CCL_ZE_IPC_EXCHANGE=sockets
export ZE_FLAT_DEVICE_HIERARCHY=FLAT

# this ulimit is generally set because of how PyTorch
# parallel data loader workers are implemented
ulimit -n 60000
srun python xpu_ddp_slurm.py
