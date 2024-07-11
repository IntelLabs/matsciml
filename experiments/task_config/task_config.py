from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytorch_lightning as pl

from matsciml.common.registry import registry
from matsciml.models.base import MultiTaskLitModule

from experiments.datasets import available_data
from experiments.models import available_models
from experiments.utils.utils import instantiate_arg_dict, update_arg_dict


def setup_task(config: dict[str, Any]) -> pl.LightningModule:
    model = config["model"]
    data_task_dict = config["dataset"]
    model = instantiate_arg_dict(deepcopy(available_models[model]))
    model = update_arg_dict("model", model, config["cli_args"])
    configured_tasks = []
    data_task_list = []
    for dataset_name, tasks in data_task_dict.items():
        dset_args = deepcopy(available_data[dataset_name])
        dset_args = update_arg_dict("dataset", dset_args, config["cli_args"])
        for task in tasks:
            task_class = registry.get_task_class(task["task"])
            task_args = deepcopy(available_models["generic"])
            task_args.update(model)
            task_args.update({"task_keys": task["targets"]})
            additonal_task_args = dset_args.get("task_args", None)
            if additonal_task_args is not None:
                task_args.update(additonal_task_args)
            configured_task = task_class(**task_args)
            configured_tasks.append(configured_task)
            data_task_list.append(
                [available_data[dataset_name]["dataset"], configured_task]
            )

    if len(configured_tasks) > 1:
        task = MultiTaskLitModule(*data_task_list)
    else:
        task = configured_tasks[0]
    return task
