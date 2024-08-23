from __future__ import annotations

import pytorch_lightning as pl
from pytorch_lightning.callbacks import StochasticWeightAveraging

from matsciml.datasets.transforms import DistancesTransform, PointCloudToGraphTransform
from matsciml.lightning.data_utils import MatSciMLDataModule
from matsciml.lightning.callbacks import ExponentialMovingAverageCallback
from matsciml.models import SchNet
from matsciml.models.base import ScalarRegressionTask

"""
This script demonstrates how to use the EMA and SWA callbacks,
which are pretty necessary for models such as MACE.

EMA is implemented within ``matsciml`` using native PyTorch, whereas SWA uses
the PyTorch Lightning implementation.
"""

# construct a scalar regression task with SchNet encoder
task = ScalarRegressionTask(
    encoder_class=SchNet,
    # kwargs to be passed into the creation of SchNet model
    encoder_kwargs={
        "encoder_only": True,
        "hidden_feats": [128, 128, 128],
        "atom_embedding_dim": 128,
    },
    output_kwargs={"lazy": False, "hidden_dim": 128, "input_dim": 128},
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

# run several epochs with a limited number of train batches
# to make sure nothing breaks between updates
trainer = pl.Trainer(
    max_epochs=5,
    limit_train_batches=10,
    logger=False,
    enable_checkpointing=False,
    callbacks=[
        StochasticWeightAveraging(swa_lrs=1e-2, swa_epoch_start=1),
        ExponentialMovingAverageCallback(decay=0.99),
    ],
)
trainer.fit(task, datamodule=dm)
