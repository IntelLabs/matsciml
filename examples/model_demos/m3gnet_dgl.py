from __future__ import annotations

import pytorch_lightning as pl

from matsciml.datasets.utils import element_types
from matsciml.lightning.data_utils import MatSciMLDataModule
from matsciml.models import M3GNet
from matsciml.models.base import ScalarRegressionTask

# construct a scalar regression task with SchNet encoder
task = ScalarRegressionTask(
    encoder_class=M3GNet,
    encoder_kwargs={
        "element_types": element_types(),
    },
    task_keys=["energy_total"],
)

dm = MatSciMLDataModule.from_devset(
    "M3GNomadDataset",
    num_workers=0,
    batch_size=4,
)

# run a quick training loop
trainer = pl.Trainer(fast_dev_run=10)
trainer.fit(task, datamodule=dm)
