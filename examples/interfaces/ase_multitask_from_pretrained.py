from __future__ import annotations

from ase import Atoms, units
from ase.md.verlet import VelocityVerlet

from matsciml.interfaces.ase import MatSciMLCalculator
from matsciml.interfaces.ase.multitask import AverageTasks
from matsciml.datasets.transforms import (
    PeriodicPropertiesTransform,
    PointCloudToGraphTransform,
)

"""
Demonstrates setting up a calculator from a pretrained
multitask/multidata module, using an averaging strategy to
merge output heads.

As an example, if we trained force regression on multiple datasets
simultaneously, we would average the outputs from each "dataset",
similar to an ensemble prediction without any special weighting.

Substitute 'model.ckpt' for the path to a checkpoint file.
"""

d = 2.9
L = 10.0

atoms = Atoms("C", positions=[[0, L / 2, L / 2]], cell=[d, L, L], pbc=[1, 0, 0])

calc = MatSciMLCalculator.from_pretrained_force_regression(
    "model.ckpt",
    transforms=[
        PeriodicPropertiesTransform(6.0, True),
        PointCloudToGraphTransform("pyg"),
    ],
    multitask_strategy=AverageTasks(),  # also can be specified as a string
)
# set the calculator to matsciml
atoms.calc = calc
# run the simulation for 100 timesteps, with 5 femtosecond timesteps
dyn = VelocityVerlet(atoms, timestep=5 * units.fs, logfile="md.log")
dyn.run(100)
