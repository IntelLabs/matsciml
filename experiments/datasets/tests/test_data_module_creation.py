from __future__ import annotations


import pytest

import matsciml
import matsciml.datasets.transforms  # noqa: F401
from experiments.datasets.data_module_config import setup_datamodule

single_task = {"dataset": {"oqmd": {"sr": ["energy"]}}}
multi_task = {"dataset": {"s2ef": {"sr": "energy", "fr": ["force"]}}}
multi_data = {
    "dataset": {
        "s2ef": {"sr": "energy", "fr": ["force"]},
        "is2re": {"sr": ["energy_init"]},
    }
}


@pytest.mark.parametrize("task_dict", [single_task, multi_task, multi_data])
def test_task_setup(task_dict):
    setup_datamodule(run_type="debug", model="m3gnet_dgl", data_task_dict=task_dict)
