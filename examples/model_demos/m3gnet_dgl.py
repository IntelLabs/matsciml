from __future__ import annotations

import lightning.pytorch as pl

from matsciml.datasets.utils import element_types
from matsciml.lightning.data_utils import MatSciMLDataModule
from matsciml.models import M3GNet
from matsciml.models.base import ScalarRegressionTask

from matsciml.datasets.transforms import MGLDataTransform
from matsciml.datasets.transforms import (
    PeriodicPropertiesTransform,
    PointCloudToGraphTransform,
)


# construct a scalar regression task with SchNet encoder
task = ScalarRegressionTask(
    encoder_class=M3GNet,
    encoder_kwargs={"element_types": element_types(), "return_all_layer_output": True},
    output_kwargs={"lazy": False, "input_dim": 64, "hidden_dim": 64},
    task_keys=["energy_total"],
)

dm = MatSciMLDataModule.from_devset(
    "NomadDataset",
    dset_kwargs={
        "transforms": [
            PeriodicPropertiesTransform(cutoff_radius=6.5),
            PointCloudToGraphTransform(backend="dgl"),
            MGLDataTransform(),
        ]
    },
    num_workers=0,
    batch_size=4,
)

# run a quick training loop
trainer = pl.Trainer(fast_dev_run=10)
trainer.fit(task, datamodule=dm)
