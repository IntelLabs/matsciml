from __future__ import annotations

import pytorch_lightning as pl

from matsciml.datasets.transforms import (
    PeriodicPropertiesTransform,
    PointCloudToGraphTransform,
)
from matsciml.datasets.utils import element_types
from matsciml.lightning.data_utils import MatSciMLDataModule
from matsciml.models import TensorNet
from matsciml.models.base import ScalarRegressionTask

# construct a scalar regression task with SchNet encoder
task = ScalarRegressionTask(
    encoder_class=TensorNet,
    encoder_kwargs={
        "element_types": element_types(),
    },
    output_kwargs={"lazy": False, "input_dim": 64, "hidden_dim": 64},
    task_keys=["energy_total"],
)

dm = MatSciMLDataModule.from_devset(
    "NomadDataset",
    dset_kwargs={
        "transforms": [
            PeriodicPropertiesTransform(cutoff_radius=6.5, adaptive_cutoff=True),
            PointCloudToGraphTransform(
                "dgl",
                cutoff_dist=20.0,
                node_keys=["pos", "atomic_numbers"],
            ),
        ],
    },
    num_workers=0,
    batch_size=4,
)

# run a quick training loop
trainer = pl.Trainer(fast_dev_run=10)
trainer.fit(task, datamodule=dm)
