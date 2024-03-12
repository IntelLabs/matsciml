from __future__ import annotations

import pytorch_lightning as pl

from matsciml.datasets.transforms import (
    PointCloudToGraphTransform,
    PeriodicPropertiesTransform,
)
from matsciml.lightning.data_utils import MatSciMLDataModule
from matsciml.models import GraphConvModel
from matsciml.models.base import ScalarRegressionTask

# configure a simple model for testing
model = GraphConvModel(100, 128, encoder_only=True)
task = ScalarRegressionTask(model, task_keys=["band_gap_ind"])

# configure alexandria devset
dm = MatSciMLDataModule.from_devset(
    "AlexandriaDataset",
    dset_kwargs={
        "transforms": [
            PeriodicPropertiesTransform(10.0, adaptive_cutoff=True),
            PointCloudToGraphTransform("dgl", cutoff_dist=10.0),
        ]
    },
)

# run 10 steps for funsies
trainer = pl.Trainer(fast_dev_run=10)

trainer.fit(task, datamodule=dm)
