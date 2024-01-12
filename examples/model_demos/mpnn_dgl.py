from __future__ import annotations

import pytorch_lightning as pl

from matsciml.datasets.transforms import DistancesTransform
from matsciml.lightning.data_utils import MatSciMLDataModule
from matsciml.models import MPNN
from matsciml.models.base import ScalarRegressionTask

# construct a scalar regression task with SchNet encoder
task = ScalarRegressionTask(
    encoder_class=MPNN,
    encoder_kwargs={
        "encoder_only": True,
        "atom_embedding_dim": 8,
        "node_out_dim": 16,
    },
    task_keys=["energy_relaxed"],
    output_kwargs={"lazy": False, "input_dim": 16, "hidden_dim": 16},
)
# MPNN expects edge features corresponding to atom-atom distances
dm = MatSciMLDataModule.from_devset(
    "IS2REDataset",
    dset_kwargs={"transforms": [DistancesTransform()]},
)

# run a quick training loop
trainer = pl.Trainer(fast_dev_run=10)
trainer.fit(task, datamodule=dm)
