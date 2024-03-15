from __future__ import annotations

import pytorch_lightning as pl
from torch import nn
from e3nn.o3 import Irreps
from mace.modules.blocks import RealAgnosticInteractionBlock

from matsciml.datasets.transforms import (
    PointCloudToGraphTransform,
    PeriodicPropertiesTransform,
)
from matsciml.lightning.data_utils import MatSciMLDataModule
from matsciml.models.base import ScalarRegressionTask
from matsciml.models.pyg.mace import MACEWrapper


"""
This example script runs through a fast development run of the IS2RE devset
in combination with a PyG implementation of EGNN.
"""

# construct IS2RE relaxed energy regression with PyG implementation of E(n)-GNN
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
    task_keys=["energy_relaxed"],
)
# matsciml devset for OCP are serialized with DGL - this transform goes between the two frameworks
dm = MatSciMLDataModule.from_devset(
    "IS2REDataset",
    dset_kwargs={
        "transforms": [
            PeriodicPropertiesTransform(6.0, adaptive_cutoff=True),
            PointCloudToGraphTransform(
                "pyg",
                node_keys=["pos", "atomic_numbers"],
            ),
        ],
    },
)

# run a quick training loop
trainer = pl.Trainer(fast_dev_run=10)
trainer.fit(task, datamodule=dm)
