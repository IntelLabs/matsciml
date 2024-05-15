from __future__ import annotations

from ase import Atoms, units
from ase.md.verlet import VelocityVerlet

from matsciml.interfaces.ase import MatSciMLCalculator
from matsciml.datasets.transforms import (
    PeriodicPropertiesTransform,
    PointCloudToGraphTransform,
)

"""
Demonstrates setting up a calculator from a pretrained
`ForceRegressionTask` checkpoint.

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
)
# set the calculator to matsciml
atoms.calc = calc
# run the simulation for 100 timesteps, with 5 femtosecond timesteps
dyn = VelocityVerlet(atoms, timestep=5 * units.fs, logfile="md.log")
dyn.run(100)
