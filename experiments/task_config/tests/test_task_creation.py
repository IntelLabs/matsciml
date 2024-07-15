from __future__ import annotations

from typing import Any

import pytest
from experiments.task_config.task_config import (
    setup_task,
)
from experiments.utils.utils import instantiate_arg_dict

import matsciml
import matsciml.datasets.transforms  # noqa: F401


single_task = {
    "model": "egnn_dgl",
    "dataset": {"oqmd": [{"task": "ScalarRegressionTask", "targets": ["band_gap"]}]},
}
multi_task = {
    "dataset": {
        "s2ef": [
            {"task": "ScalarRegressionTask", "targets": ["energy"]},
            {"task": "ForceRegressionTask", "targets": ["force"]},
        ]
    }
}
multi_data = {
    "model": "faenet_pyg",
    "dataset": {
        "oqmd": [{"task": "ScalarRegressionTask", "targets": ["energy"]}],
        "is2re": [
            {
                "task": "ScalarRegressionTask",
                "targets": ["energy_init", "energy_relaxed"],
            }
        ],
    },
}


@pytest.fixture
def test_build_model() -> dict[str, Any]:
    input_dict = {
        "encoder_class": {"class_path": "matsciml.models.M3GNet"},
        "encoder_kwargs": {
            "element_types": {"class_path": "matsciml.datasets.utils.element_types"},
            "return_all_layer_output": True,
        },
        "output_kwargs": {"lazy": False, "input_dim": 64, "hidden_dim": 64},
        "transforms": [
            {
                "class_path": "matsciml.datasets.transforms.PeriodicPropertiesTransform",
                "init_args": [{"cutoff_radius": 6.5}, {"adaptive_cutoff": True}],
            },
            {
                "class_path": "matsciml.datasets.transforms.PointCloudToGraphTransform",
                "init_args": [
                    {"backend": "dgl"},
                    {"cutoff_dist": 20.0},
                    {"node_keys": ["pos", "atomic_numbers"]},
                ],
            },
            {"class_path": "matsciml.datasets.transforms.MGLDataTransform"},
        ],
    }

    output = instantiate_arg_dict(input_dict)
    assert isinstance(
        output["transforms"][0],
        matsciml.datasets.transforms.PeriodicPropertiesTransform,
    )
    assert isinstance(
        output["transforms"][1],
        matsciml.datasets.transforms.PointCloudToGraphTransform,
    )
    return output


@pytest.mark.dependency(depends=["test_build_model"])
@pytest.mark.parametrize("task_dict", [single_task, multi_task, multi_data])
def test_task_setup(task_dict):
    other_args = {"run_type": "debug", "model": "m3gnet_dgl", "cli_args": None}
    task_dict.update(other_args)
    setup_task(config=task_dict)
