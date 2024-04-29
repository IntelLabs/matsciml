from __future__ import annotations


import pytorch_lightning as pl
from torch import nn
from torch import distributed as dist
from mace.modules import RealAgnosticInteractionBlock
from e3nn.o3 import Irreps

from matsciml.datasets.transforms import (
    PointCloudToGraphTransform,
    PeriodicPropertiesTransform,
)
from matsciml.lightning.data_utils import MatSciMLDataModule

# this is needed to register strategy and accelerator
from matsciml.lightning import xpu  # noqa: F401
from matsciml.lightning.ddp import MPIDDPStrategy, MPIEnvironment
from matsciml.models.base import ScalarRegressionTask
from matsciml.models.pyg.mace.wrapper.model import MACEWrapper

"""
This script demonstrates how to dispatch distributed data
parallel training for MACE, using the Alexandria dataset
while manually spawning parallel processes with `mpirun`.

The vast majority of the code remains the same as you would
for single instance training; the changes:

- We use the `MPIDDPStrategy` class in matsciml, which under the hood,
  relies on `MPIEnvironment` to read information about world size,
  ranks, etc. from Intel MPI environment variables. We also use the
  `ccl` communication backend for this.
- The learning rate is scaled by the square root of the total number of workers
- We pass this information into `pl.Trainer`.
"""

env = MPIEnvironment()
dist.init_process_group("ccl", world_size=env.world_size(), rank=env.global_rank())

# use Slurm to manage processes; spawning and affinity
ddp = MPIDDPStrategy(
    accelerator="xpu",
    find_unused_parameters=True,
)

# get MPI information to pass to pl.Trainer later
num_devices = ddp.cluster_environment.local_world_size
num_nodes = ddp.cluster_environment.num_nodes
world_size = ddp.cluster_environment.world_size()

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
    opt_kwargs={"lr": 5e-4 * (world_size) ** 0.5},  # scale learning rate by workers
)

# configure data module, pointing to train and validation
# splits that are on disk
dm = MatSciMLDataModule.from_devset(
    "AlexandriaDataset",
    dset_kwargs={
        "transforms": [
            PeriodicPropertiesTransform(6.0, True),
            PointCloudToGraphTransform(
                "pyg",
                node_keys=["pos", "atomic_numbers"],
            ),
        ],
    },
    num_workers=2,
    batch_size=8,
)

# run a quick debug loop to make sure things are working
trainer = pl.Trainer(
    strategy=ddp, devices=num_devices, num_nodes=num_nodes, fast_dev_run=10
)
trainer.fit(task, datamodule=dm)
