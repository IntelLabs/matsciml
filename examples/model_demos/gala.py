from __future__ import annotations

import pytorch_lightning as pl

from matsciml.datasets.transforms import DistancesTransform, PointCloudToGraphTransform
from matsciml.lightning.data_utils import MatSciMLDataModule
from matsciml.models import SchNet
from matsciml.models.base import ScalarRegressionTask

# construct a scalar regression task with SchNet encoder
task = ScalarRegressionTask(
    encoder_class=SchNet,
    # kwargs to be passed into the creation of SchNet model
    encoder_kwargs={
        "encoder_only": True,
        "hidden_feats": [128, 128, 128],
        "atom_embedding_dim": 128,
    },
    # which keys to use as targets
    task_keys=["energy_relaxed"],
)
# Use IS2RE devset to test workflow
# SchNet uses RBFs, and expects edge features corresponding to atom-atom distances
dm = MatSciMLDataModule.from_devset(
    "IS2REDataset",
    dset_kwargs={
        "transforms": [
            PointCloudToGraphTransform(
                "dgl",
                cutoff_dist=20.0,
                node_keys=["pos", "atomic_numbers"],
            ),
            DistancesTransform(),
        ],
    },
)

# run a quick training loop
trainer = pl.Trainer(fast_dev_run=1000)
trainer.fit(task, datamodule=dm)
