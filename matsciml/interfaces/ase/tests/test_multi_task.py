from __future__ import annotations

import pytest
import torch

from matsciml.models.pyg import EGNN
from matsciml.models.base import (
    MultiTaskLitModule,
    ScalarRegressionTask,
    ForceRegressionTask,
)
from matsciml.interfaces.ase import multitask as mt


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
                encoder_class=EGNN, encoder_kwargs=egnn_args, task_keys=["energy"]
            ),
        ),
        (
            "IS2REDataset",
            ForceRegressionTask(encoder_class=EGNN, encoder_kwargs=egnn_args),
        ),
        (
            "S2EFDataset",
            ForceRegressionTask(encoder_class=EGNN, encoder_kwargs=egnn_args),
        ),
        (
            "AlexandriaDataset",
            ForceRegressionTask(encoder_class=EGNN, encoder_kwargs=egnn_args),
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
