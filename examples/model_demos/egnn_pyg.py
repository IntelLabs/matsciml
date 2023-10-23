import pytorch_lightning as pl

from matsciml.models.pyg import EGNN
from matsciml.models.base import ScalarRegressionTask
from matsciml.lightning.data_utils import MatSciMLDataModule
from matsciml.datasets.transforms import DistancesTransform

"""
This example script runs through a fast development run of the IS2RE devset
in combination with a PyG implementation of EGNN.
"""


# construct a scalar regression task with SchNet encoder
task = ScalarRegressionTask(
    encoder_class=EGNN,
    encoder_kwargs={"hidden_dim": 128, "output_dim": 64},
    task_keys=["energy_relaxed"],
)
# MPNN expects edge features corresponding to atom-atom distances
dm = MatSciMLDataModule.from_devset(
    "IS2REDataset", dset_kwargs={"transforms": [DistancesTransform()]}
)

# run a quick training loop
trainer = pl.Trainer(fast_dev_run=10)
trainer.fit(task, datamodule=dm)
