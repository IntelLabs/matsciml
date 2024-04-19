from __future__ import annotations

import pytorch_lightning as pl

from matsciml.datasets.transforms import (
    FrameAveraging,
    PointCloudToGraphTransform,
)
from matsciml.lightning.data_utils import MatSciMLDataModule
from matsciml.models.base import ForceRegressionTask
from matsciml.models.pyg import FAENet

"""
This example script runs through a fast development run of the IS2RE devset
in combination with a PyG implementation of FAENet.
"""

# construct IS2RE relaxed energy regression with PyG implementation of FAENet
task = ForceRegressionTask(
    encoder_class=FAENet,
    encoder_kwargs={
        "average_frame_embeddings": False,  # set to false for use with FA transform
        "pred_as_dict": False,
        "hidden_dim": 128,
        "out_dim": 128,
        "tag_hidden_channels": 0,
    },
     output_kwargs={"lazy": False, "input_dim": 128, "hidden_dim": 128},
    task_keys=["force"],
)

# ### matsciml devset for OCP are serialized with DGL - this transform goes between the two frameworks
dm = MatSciMLDataModule.from_devset(
    "S2EFDataset",
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
trainer = pl.Trainer(enable_checkpointing=False, logger=False, devices=1)
trainer.fit(task, datamodule=dm)