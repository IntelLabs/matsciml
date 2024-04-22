from __future__ import annotations

import pytorch_lightning as pl

from matsciml.datasets.transforms import (
    PointCloudToGraphTransform,
    PeriodicPropertiesTransform,
    NoisyPositions,
)
from matsciml.lightning.data_utils import MatSciMLDataModule
from matsciml.models.base import NodeDenoisingTask
from matsciml.models.pyg import EGNN

"""
This example script shows EGNN being used for a denoising
pretraining task, as described in:

Pre-training via denoising for molecular property prediction

by Zaidi _et al._, ICLR 2023; https://openreview.net/pdf?id=tYIMtogyee

The permutation of transforms is not fully invariant, based on how
it is currently implemented. This configuration is the recommended one,
where positions are noised _after_ generating the periodic properties;
this is to ensure that the periodic offsets are generated based on
the noise-free positions.
"""

# construct IS2RE relaxed energy regression with PyG implementation of E(n)-GNN
task = NodeDenoisingTask(
    encoder_class=EGNN,
    encoder_kwargs={"hidden_dim": 128, "output_dim": 64},
)
# set up the data module
dm = MatSciMLDataModule.from_devset(
    "AlexandriaDataset",
    dset_kwargs={
        "transforms": [
            PeriodicPropertiesTransform(6.0, True),
            NoisyPositions(
                scale=1e-3
            ),  # this sets the scale of the Gaussian noise added
            PointCloudToGraphTransform(
                "pyg",
                node_keys=[
                    "pos",
                    "noisy_pos",
                    "atomic_numbers",
                ],  # ensure noisy_pos is included for the task
            ),
        ],
    },
)

# run a quick training loop
trainer = pl.Trainer(fast_dev_run=10)
trainer.fit(task, datamodule=dm)
