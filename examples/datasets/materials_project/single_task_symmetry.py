from __future__ import annotations

import pytorch_lightning as pl
from torch.nn import LayerNorm, SiLU

from matsciml.datasets.transforms import PointCloudToGraphTransform
from matsciml.lightning.data_utils import MatSciMLDataModule
from matsciml.models import GraphConvModel
from matsciml.models.base import CrystalSymmetryClassificationTask

pl.seed_everything(21616)


model = GraphConvModel(100, 128, encoder_only=True)
task = CrystalSymmetryClassificationTask(
    model,
    output_kwargs={
        "norm": LayerNorm(128),
        "hidden_dim": 128,
        "activation": SiLU,
        "lazy": False,
        "input_dim": 128,
    },
    lr=1e-3,
)

# the base set is required because the devset does not contain symmetry labels
dm = MatSciMLDataModule(
    dataset="MaterialsProjectDataset",
    train_path="./mp-project/base/train",
    dset_kwargs={
        "transforms": [
            PointCloudToGraphTransform(
                "dgl",
                cutoff_dist=20.0,
                node_keys=["pos", "atomic_numbers"],
            ),
        ],
    },
    val_split=0.2,
    batch_size=16,
    num_workers=0,
)

trainer = pl.Trainer(max_epochs=10, enable_checkpointing=False)

trainer.fit(task, datamodule=dm)
