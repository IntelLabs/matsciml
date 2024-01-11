from __future__ import annotations

import pytorch_lightning as pl

from matsciml.lightning.data_utils import MatSciMLDataModule
from matsciml.models import GalaPotential
from matsciml.models.base import CrystalSymmetryClassificationTask

dm = MatSciMLDataModule.from_devset(
    "SyntheticPointGroupDataset",
)


task = CrystalSymmetryClassificationTask(
    encoder_class=GalaPotential,
    encoder_kwargs={"D_in": 200, "encoder_only": True, "depth": 2, "hidden_dim": 16},
)

trainer = pl.Trainer(fast_dev_run=10)
trainer.fit(task, datamodule=dm)
