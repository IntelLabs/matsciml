from __future__ import annotations

from copy import deepcopy
from typing import Any
from pathlib import Path


import lightning.pytorch as pl

from matsciml.common.registry import registry
from matsciml.models.base import MultiTaskLitModule, BaseTaskModule
from matsciml.models import multitask_from_checkpoint

from experiments.utils.configurator import configurator
from experiments.utils.utils import instantiate_arg_dict, update_arg_dict


def setup_task(config: dict[str, Any]) -> pl.LightningModule:
    model = config["model"]
    data_task_dict = config["dataset"]
    model = instantiate_arg_dict(deepcopy(configurator.models[model]))
    model = update_arg_dict("model", model, config["cli_args"])
    configured_tasks = []
    data_task_list = []
    from_checkpoint = True if "load_weights" in config else False
    for dataset_name, tasks in data_task_dict.items():
        dset_args = deepcopy(configurator.datasets[dataset_name])
        dset_args = update_arg_dict("dataset", dset_args, config["cli_args"])
        for task in tasks:
            task_class = registry.get_task_class(task["task"])
            task_args = deepcopy(configurator.models["generic"])
            task_args.update(model)
            task_args.update({"task_keys": task["targets"]})
            additonal_task_args = dset_args.get("task_args", None)
            if additonal_task_args is not None:
                task_args.update(additonal_task_args)
            configured_task = task_class if from_checkpoint else task_class(**task_args)
            configured_tasks.append(configured_task)
            data_task_list.append(
                [configurator.datasets[dataset_name]["dataset"], configured_task]
            )

    if len(configured_tasks) > 1:
        task = MultiTaskLitModule(*data_task_list)
    else:
        task = configured_tasks[0]
    if "load_weights" in config:
        task = load_from_checkpoint(task, config, task_args)
    return task


def load_from_checkpoint(
    task: BaseTaskModule, config: dict[str, Any], task_args: dict[str, Any]
) -> BaseTaskModule:
    load_config = config["load_weights"]
    ckpt = load_config["path"]
    method = load_config["method"]
    load_type = load_config["type"]
    if not isinstance(task, MultiTaskLitModule):
        if load_type == "wandb":
            # creates lightning wandb logger object and a new run
            wandb_logger = get_wandb_logger()
            artifact = Path(wandb_logger.download_artifact(ckpt))
            ckpt = artifact.joinpath("model.ckpt")
        if method == "checkpoint":
            task = task.load_from_checkpoint(ckpt)
        elif method == "pretrained":
            task = task.from_pretrained_encoder(ckpt, **task_args)
        else:
            raise Exception(
                "Unsupported method for loading checkpoint. Must be 'checkpoint' or 'pretrained'"
            )
    else:
        task = multitask_from_checkpoint(ckpt)
    return task


def get_wandb_logger():
    trainer_args = configurator.trainer
    for logger in trainer_args["loggers"]:
        if "WandbLogger" in logger["class_path"]:
            wandb_logger = instantiate_arg_dict(logger)
            return wandb_logger
    else:
        raise KeyError("WandbLogger Expected in trainer config but not found")
