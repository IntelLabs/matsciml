from __future__ import annotations

import pytest
import torch
import numpy as np
from ase import Atoms, units
from ase.md import VelocityVerlet

from matsciml.models.pyg import EGNN
from matsciml.models.base import (
    MultiTaskLitModule,
    ScalarRegressionTask,
    ForceRegressionTask,
)
from matsciml.datasets.transforms import (
    PeriodicPropertiesTransform,
    PointCloudToGraphTransform,
)
from matsciml.interfaces.ase import multitask as mt
from matsciml.interfaces.ase import MatSciMLCalculator


@pytest.fixture
def test_pbc() -> Atoms:
    pos = np.random.normal(0.0, 1.0, size=(16, 3)) * 10.0
    atomic_numbers = np.random.randint(1, 100, size=(16,))
    cell = np.eye(3).astype(float)
    return Atoms(numbers=atomic_numbers, positions=pos, cell=cell)


@pytest.fixture
def pbc_transform() -> list:
    return [PeriodicPropertiesTransform(6.0, True), PointCloudToGraphTransform("pyg")]


@pytest.fixture
def egnn_args():
    return {"hidden_dim": 32, "output_dim": 32}


@pytest.fixture
def single_data_multi_task_combo(egnn_args):
    output = {
        "IS2REDataset": {
            "ScalarRegressionTask": {"energy": torch.rand(1, 1)},
            "ForceRegressionTask": {
                "energy": torch.rand(1, 1),
                "force": torch.rand(32, 3),
            },
        }
    }
    task = MultiTaskLitModule(
        (
            "IS2REDataset",
            ScalarRegressionTask(
                encoder_class=EGNN, encoder_kwargs=egnn_args, task_keys=["energy"]
            ),
        ),
        (
            "IS2REDataset",
            ForceRegressionTask(
                encoder_class=EGNN,
                encoder_kwargs=egnn_args,
                output_kwargs={"lazy": False, "input_dim": 32},
            ),
        ),
    )
    return output, task


@pytest.fixture
def multi_data_multi_task_combo(egnn_args):
    output = {
        "IS2REDataset": {
            "ScalarRegressionTask": {"energy": torch.rand(1, 1)},
            "ForceRegressionTask": {
                "energy": torch.rand(1, 1),
                "force": torch.rand(32, 3),
            },
        },
        "S2EFDataset": {
            "ForceRegressionTask": {
                "energy": torch.rand(1, 1),
                "force": torch.rand(32, 3),
            }
        },
        "AlexandriaDataset": {
            "ForceRegressionTask": {
                "energy": torch.rand(1, 1),
                "force": torch.rand(32, 3),
            }
        },
    }
    task = MultiTaskLitModule(
        (
            "IS2REDataset",
            ScalarRegressionTask(
                encoder_class=EGNN,
                encoder_kwargs=egnn_args,
                task_keys=["energy"],
                output_kwargs={"lazy": False, "hidden_dim": 32, "input_dim": 32},
            ),
        ),
        (
            "IS2REDataset",
            ForceRegressionTask(
                encoder_class=EGNN,
                encoder_kwargs=egnn_args,
                output_kwargs={"lazy": False, "hidden_dim": 32, "input_dim": 32},
            ),
        ),
        (
            "S2EFDataset",
            ForceRegressionTask(
                encoder_class=EGNN,
                encoder_kwargs=egnn_args,
                output_kwargs={"lazy": False, "hidden_dim": 32, "input_dim": 32},
            ),
        ),
        (
            "AlexandriaDataset",
            ForceRegressionTask(
                encoder_class=EGNN,
                encoder_kwargs=egnn_args,
                output_kwargs={"lazy": False, "hidden_dim": 32, "input_dim": 32},
            ),
        ),
    )
    return output, task


def test_average_single_data(single_data_multi_task_combo):
    # unpack the fixtrure
    output, task = single_data_multi_task_combo
    strat = mt.AverageTasks()
    # test the parsing
    _, parsed_output = strat.parse_outputs(output, task)
    agg_results = strat.merge_outputs(parsed_output)
    end = strat(output, task)
    assert end
    assert agg_results
    for key in ["energy", "forces"]:
        assert key in end, f"{key} was missing from agg results"
    assert end["forces"].shape == (32, 3)


def test_average_multi_data(multi_data_multi_task_combo):
    # unpack the fixtrure
    output, task = multi_data_multi_task_combo
    strat = mt.AverageTasks()
    # test the parsing
    _, parsed_output = strat.parse_outputs(output, task)
    agg_results = strat.merge_outputs(parsed_output)
    end = strat(output, task)
    assert end
    assert agg_results
    for key in ["energy", "forces"]:
        assert key in end, f"{key} was missing from agg results"
    assert end["forces"].shape == (32, 3)


def test_calc_multi_data(
    multi_data_multi_task_combo, test_pbc: Atoms, pbc_transform: list
):
    output, task = multi_data_multi_task_combo
    strat = mt.AverageTasks()
    calc = MatSciMLCalculator(task, multitask_strategy=strat, transforms=pbc_transform)
    atoms = test_pbc.copy()
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    assert np.isfinite(energy)
    forces = atoms.get_forces()
    assert np.isfinite(forces).all()
    dyn = VelocityVerlet(atoms, timestep=5 * units.fs, logfile="md.log")
    dyn.run(3)
