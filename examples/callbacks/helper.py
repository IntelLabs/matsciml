from __future__ import annotations

import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger

from matsciml.datasets.transforms import DistancesTransform, PointCloudToGraphTransform
from matsciml.lightning.data_utils import MatSciMLDataModule
from matsciml.lightning.callbacks import TrainingHelperCallback
from matsciml.models import SchNet
from matsciml.models.base import ScalarRegressionTask

"""
This script demonstrates the use of the ``TrainingHelperCallback``
callback. The purpose of this callback is to provide some
helpful heuristics into the training process by identifying
some common issues like unused weights, small gradients,
and oversmoothed embeddings.
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
    # which keys to use as targets
    task_keys=["energy_relaxed"],
    log_embeddings=True,
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

# tensorboard logging if working purely locally
# logger = TensorBoardLogger("./")
logger = WandbLogger(
    name="helper-callback", offline=False, project="matsciml", log_model="all"
)

# run a quick training loop
trainer = pl.Trainer(max_epochs=10, logger=logger, callbacks=[TrainingHelperCallback()])
trainer.fit(task, datamodule=dm)
