from __future__ import annotations

from ase import Atoms, units
from ase.md.verlet import VelocityVerlet
import pytorch_lightning as pl

from matsciml.lightning import MatSciMLDataModule
from matsciml.models.base import ForceRegressionTask
from matsciml.models.pyg import EGNN
from matsciml.interfaces.ase import MatSciMLCalculator
from matsciml.datasets.transforms import (
    PeriodicPropertiesTransform,
    PointCloudToGraphTransform,
)

"""
Demonstrates setting up a calculator from a `ForceRegressionTask`
trained from scratch - this is unlikely be the way you would actually
do this, but just demonstrates how the workflow is composed together
in a single file.
"""

task = ForceRegressionTask(
    encoder_class=EGNN,
    encoder_kwargs={"hidden_dim": 32, "output_dim": 32},
    output_kwargs={"lazy": False, "input_dim": 32, "hidden_dim": 32, "num_hidden": 3},
)

transforms = [
    PeriodicPropertiesTransform(6.0, True),
    PointCloudToGraphTransform("pyg"),
]

dm = MatSciMLDataModule.from_devset(
    "LiPSDataset", batch_size=8, num_workers=0, dset_kwargs={"transforms": transforms}
)

# run the training loop
trainer = pl.Trainer(
    fast_dev_run=10,
    logger=False,
    enable_checkpointing=False,
)
trainer.fit(task, datamodule=dm)

# put it into eval for inference
task = task.eval()

# get a random frame from LiPS to do the propagation
frame = dm.dataset.__getitem__(52)
graph = frame["graph"]
atoms = Atoms(
    positions=graph["pos"].numpy(),
    cell=frame["cell"].numpy().squeeze(),
    numbers=graph["atomic_numbers"].numpy(),
)

# instantiate calculator using the trained model
# reuse the same transforms as with the data module
calc = MatSciMLCalculator(task, transforms=transforms)
# set the calculator to matsciml
atoms.calc = calc
# run the simulation for 100 timesteps, with 5 femtosecond timesteps
dyn = VelocityVerlet(atoms, timestep=5 * units.fs, logfile="md.log")
dyn.run(100)
