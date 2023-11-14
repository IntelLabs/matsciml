from __future__ import annotations

import pytorch_lightning as pl

from matsciml.datasets.transforms import FrameAveraging
from matsciml.datasets.transforms import GraphToGraphTransform
from matsciml.datasets.transforms import PointCloudToGraphTransform
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
        "pred_as_dict": False,
        "hidden_dim": 128,
        "output_dim": 64,
        "tag_hidden_channels": 0,
    },
    # output_kwargs={"lazy": False, "input_dim": 64},
    task_keys=["energy_relaxed"],
    # task_keys=["band_gap"]
)
### matsciml devset for OCP are serialized with DGL - this transform goes between the two frameworks
dm = MatSciMLDataModule.from_devset(
    "IS2REDataset",
    dset_kwargs={
        "transforms": [
            GraphToGraphTransform("pyg"),
            FrameAveraging(frame_averaging="3D", fa_method="stochastic"),
        ],
    },
)


# dm = MatSciMLDataModule.from_devset(
#     "MaterialsProjectDataset",
#     dset_kwargs={
#         "transforms": [
#             PointCloudToGraphTransform(
#                 "dgl", cutoff_dist=20.0, node_keys=["pos", "atomic_numbers"]
#             ),
#             GraphToGraphTransform("pyg"),
#             FrameAveraging(frame_averaging="3D", fa_method="stochastic"),
#         ],
#     },
# )

# run a quick training loop
trainer = pl.Trainer(fast_dev_run=10)
trainer.fit(task, datamodule=dm)
