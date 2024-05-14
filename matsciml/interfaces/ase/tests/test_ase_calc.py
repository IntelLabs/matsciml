from __future__ import annotations

import pytest
import numpy as np
from ase import Atoms
import torch

from matsciml.datasets.transforms import (
    PeriodicPropertiesTransform,
    PointCloudToGraphTransform,
)
from matsciml.interfaces.ase import MatSciMLCalculator
from matsciml.models.base import (
    ForceRegressionTask,
)
from matsciml.models.pyg import EGNN

np.random.seed(21516136)


@pytest.fixture
def test_molecule() -> Atoms:
    pos = np.random.normal(0.0, 1.0, size=(10, 3))
    atomic_numbers = np.random.randint(1, 100, size=(10,))
    return Atoms(numbers=atomic_numbers, positions=pos)


@pytest.fixture
def test_pbc() -> Atoms:
    pos = np.random.normal(0.0, 1.0, size=(16, 3))
    atomic_numbers = np.random.randint(1, 100, size=(16,))
    cell = np.eye(3).astype(float)
    return Atoms(numbers=atomic_numbers, positions=pos, cell=cell)


@pytest.fixture
def pbc_transform() -> list:
    return [PeriodicPropertiesTransform(6.0, True), PointCloudToGraphTransform("pyg")]


@pytest.fixture
def egnn_config():
    return {"hidden_dim": 32, "output_dim": 32}


@pytest.mark.parametrize("dtype", [torch.float, torch.float64, torch.bfloat16])
def test_egnn_energy_forces(
    dtype: torch.dtype, egnn_config: dict, test_pbc: Atoms, pbc_transform: list
):
    task = ForceRegressionTask(
        encoder_class=EGNN, encoder_kwargs=egnn_config, output_kwargs={"hidden_dim": 32}
    ).to(dtype)
    calc = MatSciMLCalculator(task, transforms=pbc_transform)
    atoms = test_pbc.copy()
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    assert np.isfinite(energy)
    forces = atoms.get_forces()
    assert np.isfinite(forces).all()
