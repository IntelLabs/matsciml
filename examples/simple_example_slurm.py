# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: MIT License

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.plugins.environments import SLURMEnvironment
from pytorch_lightning.strategies import DDPStrategy
import dgl
import torch

"""
simple_example_slurm.py

This script launches the training run on a Slurm managed HPC cluster.
The role of the `SLURMEnvironment` is to inform the communication backend
on which environment variables to read worker information (e.g. rank, world size)
from. In addition, we can also enable automatic requeuing of the training
job---this will restart the training from prior to hitting the wall time limit.

This mode of operation will also defer worker launching to Slurm: instead of
executing the script with an `mpirun` call, the workers will be created based
on the Slurm configuration; number of nodes and the number of workers per node.
This depends on Slurm being configured to be able to do so, and your mileage
may vary: if in doubt, it is recommended to launch manually using `mpirun` particularly
for CPU-based jobs.

For this to script to work, you will need to build PyTorch from source with MPI
support: consult the main README for more instructions.

Run this by submitting the Slurm script:
    `sbatch slurm_submit.sh`
"""

try:
    from ocpmodels.datasets import is2re_devset
    from ocpmodels.lightning.data_utils import IS2REDGLDataModule
    from ocpmodels.models import DimeNetPP, IS2RELitModule
except ImportError:
    import sys, os

    dir_path = os.path.dirname(os.path.realpath(__file__))
    sys.path.append("{}/../".format(dir_path))

    from ocpmodels.datasets import is2re_devset
    from ocpmodels.lightning.data_utils import IS2REDGLDataModule
    from ocpmodels.models import DimeNetPP, IS2RELitModule


### Hardcoded settings for testing purposes
SEED = 42
BATCH_SIZE = 16
# change this path to where your data resides (folder containing LMDB files)
DATA_PATH = is2re_devset

# this sets the random seeds for all (i.e. torch, random, numpy) except DGL
pl.seed_everything(SEED)
dgl.seed(SEED)

# default model configuration for DimeNet++
model_config = {
    "emb_size": 128,
    "out_emb_size": 256,
    "int_emb_size": 64,
    "basis_emb_size": 8,
    "num_blocks": 3,
    "num_spherical": 7,
    "num_radial": 6,
    "cutoff": 10.0,
    "envelope_exponent": 5.0,
    "activation": torch.nn.SiLU,
}

# use default settings for DimeNet++
dpp = DimeNetPP(**model_config)
model = IS2RELitModule(dpp, lr=1e-3, gamma=0.1)

data_module = IS2REDGLDataModule(DATA_PATH, batch_size=BATCH_SIZE, num_workers=0)

# configure PyTorch Lightning to use SLURM using MPI as the communications backend
environment = SLURMEnvironment(auto_requeue=False)
# "mpi" can also be swapped for "ccl" for oneCCL optimized communications
strategy = DDPStrategy(process_group_backend="mpi", cluster_environment=environment)
# create logger
logger = CSVLogger("lightning_logs", name="DimeNetPP")

# configure trainer: accelerator can also be GPU if desired
trainer = pl.Trainer(
    strategy=strategy,
    accelerator="cpu",
    logger=logger,
    max_epochs=5,  # run 5 epochs of 10 steps each
    log_every_n_steps=50,
)

# run the training procedure
trainer.fit(model, datamodule=data_module)
