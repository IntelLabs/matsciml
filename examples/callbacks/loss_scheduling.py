from __future__ import annotations

import pytorch_lightning as pl

from matsciml.datasets.transforms import (
    PeriodicPropertiesTransform,
    DistancesTransform,
    PointCloudToGraphTransform,
)
from matsciml.lightning.data_utils import MatSciMLDataModule
from matsciml.models import SchNet
from matsciml.models.base import ScalarRegressionTask

from matsciml.lightning.callbacks import LossScalingScheduler
from matsciml.lightning.loss_scaling import SigmoidScalingSchedule

"""
This script demonstrates how to add loss scaling schedules
to training runs.
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
            PeriodicPropertiesTransform(6.0, True),
            PointCloudToGraphTransform(
                "dgl",
                node_keys=["pos", "atomic_numbers"],
            ),
            DistancesTransform(),
        ],
    },
)

# run several epochs with a limited number of train batches
# to make sure nothing breaks between updates
trainer = pl.Trainer(
    max_epochs=10,
    limit_train_batches=10,
    logger=False,
    enable_checkpointing=False,
    callbacks=[
        LossScalingScheduler(
            SigmoidScalingSchedule(
                "energy_relaxed",
                initial_value=10.0,  # the first value will not be this exactly
                end_value=1.0,  # but close to it, due to nature of sigmoid
                center_frac=0.5,  # this means the sigmoid flips at half the total steps
                curvature=1e-7,  # can be modified to change ramping behavior
                step_frequency="step",
            ),
            log_level="DEBUG",  # this makes it verbose, but setting it to INFO will surpress most
        )
    ],
)
trainer.fit(task, datamodule=dm)
# print out the final scaling rates
print(task.task_loss_scaling)
