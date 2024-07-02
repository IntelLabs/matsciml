from __future__ import annotations

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from matsciml.datasets.transforms import DistancesTransform, PointCloudToGraphTransform
from matsciml.lightning.data_utils import MatSciMLDataModule
from matsciml.lightning.callbacks import ModelAutocorrelation
from matsciml.models import SchNet
from matsciml.models.base import ScalarRegressionTask

"""
This script demonstrates the use of the `ModelAutocorrelation` callback.

The main utility of this callback is to monitor the degree of correlation
in model parameters and optionally gradients over a time span. The idea
is that for optimization trajectories, steps are ideally as de-correlated
as possible (at least within reason), and indeed is actually a major
assumption of Adam-like optimizers.

There is no hard coded heuristic for identifying "too much correlation"
yet, however this callback can help do the data collection for you to
develop a sense for yourself. One method for trying this out is to
set varying learning rates, and seeing how the autocorrelation spectra
are different.
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
    log_embeddings=False,
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

# tensorboard logging if working purely locally, otherwise wandb
logger = WandbLogger(
    name="helper-callback", offline=False, project="matsciml", log_model="all"
)
logger = TensorBoardLogger("./")

# run a quick training loop
trainer = pl.Trainer(max_epochs=30, logger=logger, callbacks=[ModelAutocorrelation()])
trainer.fit(task, datamodule=dm)
