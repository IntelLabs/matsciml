from __future__ import annotations

import pytorch_lightning as pl

from matsciml.datasets.transforms import (
    PeriodicPropertiesTransform,
    PointCloudToGraphTransform,
)
from matsciml.lightning import MatSciMLDataModule
from matsciml.models import ScalarRegressionTask
from matsciml.models.pyg import EGNN


def test_egnn_end_to_end():
    """
    Test the end to end pipeline using a devset with EGNN.

    The idea is that this basically mimics an example script to
    try and maximize coverage across dataset to training, which
    is particularly useful for checking new dependencies, etc.
    """
    dm = MatSciMLDataModule.from_devset(
        "MaterialsProjectDataset",
        dset_kwargs={
            "transforms": [
                PeriodicPropertiesTransform(6.0, adaptive_cutoff=True),
                PointCloudToGraphTransform("pyg"),
            ]
        },
        batch_size=8,
    )

    # this specifies a whole lot to make sure we have coverage
    task = ScalarRegressionTask(
        encoder_class=EGNN,
        encoder_kwargs={
            "hidden_dim": 48,
            "output_dim": 32,
            "num_conv": 2,
            "num_atom_embedding": 200,
        },
        scheduler_kwargs={
            "CosineAnnealingLR": {
                "T_max": 5,
                "eta_min": 1e-7,
            }
        },
        lr=1e-3,
        weight_decay=0.0,
        output_kwargs={
            "lazy": False,
            "hidden_dim": 48,
            "input_dim": 48,
            "dropout": 0.2,
            "num_hidden": 2,
        },
        task_keys=["band_gap"],
    )

    trainer = pl.Trainer(fast_dev_run=5)
    trainer.fit(task, datamodule=dm)
