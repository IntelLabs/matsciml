from __future__ import annotations

import lightning.pytorch as pl

from matsciml.datasets.transforms import (
    PointCloudToGraphTransform,
    PeriodicPropertiesTransform,
)
from matsciml.lightning.callbacks import CodeCarbonCallback
from matsciml.lightning.data_utils import MatSciMLDataModule
from matsciml.models.base import ScalarRegressionTask
from matsciml.models.pyg import EGNN

"""
This example script runs through a fast development run of the IS2RE devset
in combination with a PyG implementation of EGNN.
"""

# construct IS2RE relaxed energy regression with PyG implementation of E(n)-GNN
task = ScalarRegressionTask(
    encoder_class=EGNN,
    encoder_kwargs={"hidden_dim": 128, "output_dim": 64},
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
trainer = pl.Trainer(
    max_epochs=3,
    callbacks=[CodeCarbonCallback(country_iso_code="USA", measure_power_secs=1)],
)
trainer.fit(task, datamodule=dm)
