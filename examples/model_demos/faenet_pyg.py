from __future__ import annotations

import pytorch_lightning as pl

from matsciml.datasets.transforms import (
    FrameAveraging,
    GraphToGraphTransform,
    PointCloudToGraphTransform,
    UnitCellCalculator,
)
from matsciml.lightning.data_utils import MatSciMLDataModule
from matsciml.models.base import ScalarRegressionTask
from matsciml.models.pyg import FAENet

"""
This example script runs through a fast development run of the IS2RE devset
in combination with a PyG implementation of FAENet.
"""

# construct IS2RE relaxed energy regression with PyG implementation of FAENet
task = ScalarRegressionTask(
    encoder_class=FAENet,
    encoder_kwargs={
        "average_frame_embeddings": True,
        "pred_as_dict": False,
        "hidden_dim": 128,
        "out_dim": 64,
        "tag_hidden_channels": 0,
    },
    output_kwargs={"lazy": False, "input_dim": 64, "hidden_dim": 64},
    task_keys=["energy_relaxed"],
)

# ### matsciml devset for OCP are serialized with DGL - this transform goes between the two frameworks
dm = MatSciMLDataModule.from_devset(
    "IS2REDataset",
    dset_kwargs={
        "transforms": [
            PointCloudToGraphTransform(
                "pyg",
                cutoff_dist=20.0,
                node_keys=["pos", "atomic_numbers"],
            ),
            FrameAveraging(frame_averaging="3D", fa_method="stochastic"),
        ],
    },
)


# run a quick training loop
trainer = pl.Trainer(fast_dev_run=10)
trainer.fit(task, datamodule=dm)


########################################################################################
########################################################################################


# construct Materials Project band gap regression with PyG implementation of FAENet
task = ScalarRegressionTask(
    encoder_class=FAENet,
    encoder_kwargs={
        "pred_as_dict": False,
        "hidden_dim": 128,
        "out_dim": 64,
        "tag_hidden_channels": 0,
        "input_dim": 128,
    },
    output_kwargs={"lazy": False, "input_dim": 64, "hidden_dim": 64},
    task_keys=["band_gap"],
)

dm = MatSciMLDataModule.from_devset(
    "MaterialsProjectDataset",
    dset_kwargs={
        "transforms": [
            UnitCellCalculator(),
            PointCloudToGraphTransform(
                "pyg",
                cutoff_dist=20.0,
                node_keys=["pos", "atomic_numbers"],
            ),
            FrameAveraging(frame_averaging="3D", fa_method="stochastic"),
        ],
    },
)

# run a quick training loop
trainer = pl.Trainer(fast_dev_run=10)
trainer.fit(task, datamodule=dm)
