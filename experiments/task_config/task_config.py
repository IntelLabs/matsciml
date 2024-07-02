from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytorch_lightning as pl


from matsciml.models.base import (
    BinaryClassificationTask,
    CrystalSymmetryClassificationTask,
    ForceRegressionTask,
    GradFreeForceRegressionTask,
    MultiTaskLitModule,
    ScalarRegressionTask,
)

from experiments.datasets import available_data
from experiments.models import available_models
from experiments.utils.utils import instantiate_arg_dict, update_arg_dict


task_map = {
    "sr": ScalarRegressionTask,
    "scalar_regression": ScalarRegressionTask,
    "fr": ForceRegressionTask,
    "force_regression": ForceRegressionTask,
    "bc": BinaryClassificationTask,
    "binary_classification": BinaryClassificationTask,
    "csc": CrystalSymmetryClassificationTask,
    "crystal_symmetry_classification": CrystalSymmetryClassificationTask,
    "gffr": GradFreeForceRegressionTask,
    "grad_free_force_regression": GradFreeForceRegressionTask,
}


def setup_task(config: dict[str, Any]) -> pl.LightningModule:
    model = config["model"]
    data_task_dict = config["dataset"]
    model = instantiate_arg_dict(deepcopy(available_models[model]))
    model = update_arg_dict("model", model, config["cli_args"])
    tasks = []
    data_task_list = []
    for dataset_name, task_dict in data_task_dict.items():
        dset_args = deepcopy(available_data[dataset_name])
        dset_args = update_arg_dict("dataset", dset_args, config["cli_args"])
        for task_type, task_keys in task_dict.items():
            task_args = deepcopy(available_models["generic"])
            task_args.update(model)
            task_args.update({"task_keys": task_keys})
            additonal_task_args = dset_args.get("task_args", None)
            if additonal_task_args is not None:
                task_args.update(additonal_task_args)
            task = task_map[task_type](**task_args)
            tasks.append(task)
            data_task_list.append([available_data[dataset_name]["dataset"], task])
    if len(tasks) > 1:
        task = MultiTaskLitModule(*data_task_list)
    else:
        task = tasks[0]
    return task
