from __future__ import annotations

import pytorch_lightning as pl
from torch.nn import LayerNorm, SiLU

from matsciml.datasets.transforms import PointCloudToGraphTransform
from matsciml.lightning.data_utils import MatSciMLDataModule
from matsciml.models import GraphConvModel
from matsciml.models.base import ScalarRegressionTask

pl.seed_everything(21616)


model = GraphConvModel(100, 128, encoder_only=True)
task = ScalarRegressionTask(
    model,
    output_kwargs={
        "norm": LayerNorm(128),
        "hidden_dim": 128,
        "activation": SiLU,
        "lazy": False,
        "input_dim": 128,
    },
    lr=1e-3,
    task_keys=["band_gap"],
)


dm = MatSciMLDataModule(
    "MaterialsProjectDataset",
    train_path="./matsciml/datasets/materials_project/devset",
    dset_kwargs={"transforms": [PointCloudToGraphTransform("dgl", cutoff_dist=20.0)]},
    val_split=0.2,
)

trainer = pl.Trainer(fast_dev_run=10, enable_checkpointing=False)

trainer.fit(task, datamodule=dm)
