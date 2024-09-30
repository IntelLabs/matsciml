from __future__ import annotations

import pytest
import numpy as np
from ase import Atoms, units
from ase.md.verlet import VelocityVerlet

from matsciml.datasets.transforms import (
    PeriodicPropertiesTransform,
    PointCloudToGraphTransform,
)
from matsciml.interfaces.ase import MatSciMLCalculator
from matsciml.models.base import (
    ForceRegressionTask,
)
from matsciml.models.pyg import EGNN
from types import MethodType

import matgl
import torch
from matgl.ext.ase import Atoms2Graph


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


def test_egnn_energy_forces(egnn_config: dict, test_pbc: Atoms, pbc_transform: list):
    """Just get the energy and force out of a ForceRegressionTask."""
    task = ForceRegressionTask(
        encoder_class=EGNN, encoder_kwargs=egnn_config, output_kwargs={"hidden_dim": 32}
    )
    calc = MatSciMLCalculator(
        task, transforms=pbc_transform, output_map={"forces": "force"}
    )
    atoms = test_pbc.copy()
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    assert np.isfinite(energy)
    forces = atoms.get_forces()
    assert np.isfinite(forces).all()


def test_egnn_dynamics(egnn_config: dict, test_pbc: Atoms, pbc_transform: list):
    """Run a few timesteps of MD to test the workflow end-to-end."""
    task = ForceRegressionTask(
        encoder_class=EGNN, encoder_kwargs=egnn_config, output_kwargs={"hidden_dim": 32}
    )
    calc = MatSciMLCalculator(
        task, transforms=pbc_transform, output_map={"forces": "force"}
    )
    atoms = test_pbc.copy()
    atoms.calc = calc
    dyn = VelocityVerlet(atoms, timestep=5 * units.fs, logfile="md.log")
    dyn.run(3)


def test_matgl():
    matgl_model = matgl.load_model("CHGNet-MPtrj-2024.2.13-PES-11M")

    def forward(self, atoms):
        graph_converter = Atoms2Graph(
            element_types=matgl_model.model.element_types,
            cutoff=matgl_model.model.cutoff,
        )
        graph, lattice, state_feats_default = graph_converter.get_graph(atoms)
        graph.edata["pbc_offshift"] = torch.matmul(
            graph.edata["pbc_offset"], lattice[0]
        )
        graph.ndata["pos"] = graph.ndata["frac_coords"] @ lattice[0]
        state_feats = torch.tensor(state_feats_default)
        total_energies, forces, stresses, *others = self.matgl_forward(
            graph, lattice, state_feats
        )
        output = {}
        output["energy"] = total_energies
        output["forces"] = forces
        output["stress"] = stresses
        return output

    matgl_model.matgl_forward = matgl_model.forward
    matgl_model.forward = MethodType(forward, matgl_model)
    matgl_model.predict = MethodType(forward, matgl_model)

    matgl_model.predict = MethodType(forward, matgl_model)
    calc = MatSciMLCalculator(matgl_model, matsciml_model=False)
    pos = np.random.normal(0.0, 1.0, size=(10, 3))
    atomic_numbers = np.random.randint(1, 89, size=(10,))
    atoms = Atoms(numbers=atomic_numbers, positions=pos)
    atoms = atoms.copy()
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    assert np.isfinite(energy)
    forces = atoms.get_forces()
    assert np.isfinite(forces).all()
