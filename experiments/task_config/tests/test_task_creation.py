from __future__ import annotations

from typing import Any

import pytest
from experiments.task_config.task_config import (
    setup_task,
)
from experiments.utils.utils import instantiate_arg_dict

import matsciml
import matsciml.datasets.transforms  # noqa: F401


single_task = {"dataset": {"oqmd": {"sr": ["energy"]}}}
multi_task = {"dataset": {"s2ef": {"sr": "energy", "fr": ["force"]}}}
multi_data = {
    "dataset": {
        "s2ef": {"sr": "energy", "fr": ["force"]},
        "is2re": {"sr": ["energy_init"]},
    }
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
    setup_task(run_type="debug", model="m3gnet_dgl", data_task_dict=task_dict)
