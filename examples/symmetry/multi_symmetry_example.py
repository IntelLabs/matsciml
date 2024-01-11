from __future__ import annotations

import pytorch_lightning as pl

from matsciml.datasets.materials_project import (
    MaterialsProjectDataset,
    materialsproject_devset,
)
from matsciml.datasets.multi_dataset import MultiDataset
from matsciml.datasets.symmetry import SyntheticPointGroupDataset, symmetry_devset
from matsciml.datasets.transforms import PointCloudToGraphTransform
from matsciml.lightning.data_utils import MultiDataModule
from matsciml.models import GraphConvModel
from matsciml.models.base import (
    CrystalSymmetryClassificationTask,
    MultiTaskLitModule,
    ScalarRegressionTask,
)

dm = MultiDataModule(
    train_dataset=MultiDataset(
        [
            SyntheticPointGroupDataset(
                symmetry_devset,
                transforms=[PointCloudToGraphTransform("dgl", cutoff_dist=20.0)],
            ),
            MaterialsProjectDataset(
                materialsproject_devset,
                transforms=[PointCloudToGraphTransform("dgl", cutoff_dist=20.0)],
            ),
        ],
    ),
    batch_size=16,
)

sym_task = CrystalSymmetryClassificationTask(
    encoder_class=GraphConvModel,
    encoder_kwargs={"atom_embedding_dim": 200, "out_dim": 1, "encoder_only": True},
)
reg_task = ScalarRegressionTask(
    encoder_class=GraphConvModel,
    encoder_kwargs={"atom_embedding_dim": 200, "out_dim": 1, "encoder_only": True},
    task_keys=["band_gap"],
)

task = MultiTaskLitModule(
    ("SyntheticPointGroupDataset", sym_task),
    ("MaterialsProjectDataset", reg_task),
)

trainer = pl.Trainer(max_epochs=1, log_every_n_steps=5)
trainer.fit(task, datamodule=dm)
