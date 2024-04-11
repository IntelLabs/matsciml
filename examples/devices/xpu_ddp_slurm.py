from __future__ import annotations

import os

import pytorch_lightning as pl
from matsciml.datasets.transforms import (
    PointCloudToGraphTransform,
    PeriodicPropertiesTransform,
)
from matsciml.lightning.data_utils import MatSciMLDataModule
from torch import nn

# this is needed to register strategy and accelerator
from matsciml.lightning import xpu  # noqa: F401
from matsciml.models.base import ScalarRegressionTask
from matsciml.models.pyg.mace.wrapper.model import MACEWrapper
from mace.modules import RealAgnosticInteractionBlock
from e3nn.o3 import Irreps

"""
This script demonstrates how to dispatch distributed data
parallel training for MACE, using the Alexandria dataset.

The vast majority of the code remains the same as you would
for single instance training; the changes:

- We read node and device count per node from Slurm environment variables
- The learning rate is scaled by the square root of the total number of workers
- We initialize a `SLURMEnvironment` object that interfaces Lightning with
  Slurm environment variables.
- We configure a `DDPStrategy` that uses `xpu` as its accelerator, and `ccl`
  as the communication backend.
- We pass this information into `pl.Trainer`.
"""

# get device count and whatnot from Slurm
num_devices = int(os.environ["SLURM_NTASKS_PER_NODE"])
num_nodes = int(os.environ["SLURM_NNODES"])

# configure MACE architecture to perform scalar regression
# predicting the system energy, the formation energy, and the
# energy above the convex hull
task = ScalarRegressionTask(
    encoder_class=MACEWrapper,
    encoder_kwargs={
        "r_max": 6.0,
        "num_bessel": 3,
        "num_polynomial_cutoff": 3,
        "max_ell": 2,
        "interaction_cls": RealAgnosticInteractionBlock,
        "interaction_cls_first": RealAgnosticInteractionBlock,
        "num_interactions": 2,
        "atom_embedding_dim": 64,
        "MLP_irreps": Irreps("256x0e"),
        "avg_num_neighbors": 10.0,
        "correlation": 1,
        "radial_type": "bessel",
        "gate": nn.Identity(),
    },
    task_keys=["energy_total", "e_form", "e_above_hull"],
    output_kwargs={"lazy": False, "input_dim": 128, "hidden_dim": 128},
    opt_kwargs={
        "lr": 5e-4 * (num_nodes * num_devices) ** 0.5
    },  # scale learning rate by workers
)

# this needs to be set according to where the dataset is stashed
DATA_ROOT = ""

# configure data module, pointing to train and validation
# splits that are on disk
dm = MatSciMLDataModule(
    "AlexandriaDataset",
    train_path=DATA_ROOT + "/train",
    val_split=DATA_ROOT + "/val",
    dset_kwargs={
        "transforms": [
            PeriodicPropertiesTransform(6.0, True),
            PointCloudToGraphTransform(
                "pyg",
                node_keys=["pos", "atomic_numbers"],
            ),
        ],
    },
    num_workers=8,
    batch_size=32,
)

# use Slurm to manage processes; spawning and affinity
env = pl.plugins.environments.SLURMEnvironment()
ddp = pl.strategies.DDPStrategy(
    accelerator="xpu",
    cluster_environment=env,
    process_group_backend="ccl",
    find_unused_parameters=True,
)

# run a quick training loop on a single XPU device with BF16 automatic mixed precision
trainer = pl.Trainer(
    strategy=ddp,
    devices=num_devices,
    max_epochs=1,
    num_nodes=num_nodes,  # precision="bf16-mixed"
)
trainer.fit(task, datamodule=dm)
