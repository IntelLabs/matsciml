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
    "fr": ForceRegressionTask,
    "bc": BinaryClassificationTask,
    "csc": CrystalSymmetryClassificationTask,
    "gffr": GradFreeForceRegressionTask,
}


# getattr(matsciml.datasets, available_data[dataset_name]["dataset"])
def setup_task(config: dict[str, Any]) -> pl.LightningModule:
    model = config["model"]
    data_task_dict = config["dataset"]
    run_type = config["run_type"]
    model = instantiate_arg_dict(deepcopy(available_models[model]))
    model = update_arg_dict("model", model, config["cli_args"])
    tasks = []
    data_task_list = []
    for dataset_name, task_dict in data_task_dict.items():
        dset_args = deepcopy(available_data[dataset_name])
        dset_args = update_arg_dict("dataset", dset_args, config["cli_args"])
        for task_type, task_keys in task_dict.items():
            task_args = deepcopy(available_models["generic"])
            normalize_kwargs = dset_args[run_type].pop("normalize_kwargs", None)
            task_args.update(model)
            task_args.update({"task_keys": task_keys})
            task_args.update({"normalize_kwargs": normalize_kwargs})
            task = task_map[task_type](**task_args)
            tasks.append(task)
            data_task_list.append([available_data[dataset_name]["dataset"], task])

    if len(tasks) > 1:
        task = MultiTaskLitModule(*data_task_list)
    else:
        task = tasks[0]
    return task
