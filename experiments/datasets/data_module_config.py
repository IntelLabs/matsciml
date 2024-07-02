from __future__ import annotations

from copy import deepcopy
from typing import Any
import sys

import pytorch_lightning as pl

from matsciml.lightning.data_utils import (
    MatSciMLDataModule,
)
from matsciml.datasets import *  # noqa: F401

from matsciml.lightning.data_utils import MultiDataModule

from experiments.datasets import available_data
from experiments.models import available_models
from experiments.utils.utils import instantiate_arg_dict, update_arg_dict


def setup_datamodule(config: dict[str, Any]) -> pl.LightningModule:
    model = config["model"]
    data_task_dict = config["dataset"]
    run_type = config["run_type"]
    model = instantiate_arg_dict(deepcopy(available_models[model]))
    model = update_arg_dict("model", model, config["cli_args"])
    datasets = list(data_task_dict.keys())
    if len(datasets) == 1:
        dset = deepcopy(available_data[datasets[0]])
        dset = update_arg_dict("dataset", dset, config["cli_args"])
        dm_kwargs = deepcopy(available_data["generic"]["experiment"])
        dm_kwargs.update(dset[run_type])
        if run_type == "debug":
            dm = MatSciMLDataModule.from_devset(
                dataset=dset["dataset"],
                dset_kwargs={"transforms": model["transforms"]},
                **dm_kwargs,
            )
        else:
            dm = MatSciMLDataModule(
                dataset=dset["dataset"],
                dset_kwargs={"transforms": model["transforms"]},
                **dm_kwargs,
            )
    else:
        dset_list = {"train": [], "val": [], "test": []}
        for dataset in datasets:
            dset = deepcopy(available_data[dataset])
            dset = update_arg_dict("dataset", dset, config["cli_args"])
            dm_kwargs = deepcopy(available_data["generic"]["experiment"])
            dset[run_type].pop("normalize_kwargs", None)
            dm_kwargs.update(dset[run_type])
            dataset_name = dset["dataset"]
            dataset = getattr(sys.modules[__name__], dataset_name)
            model_transforms = model["transforms"]
            if run_type == "debug":
                dset_list["train"].append(
                    dataset.from_devset(transforms=model_transforms)
                )
                dset_list["val"].append(
                    dataset.from_devset(transforms=model_transforms)
                )
                dset_list["test"].append(
                    dataset.from_devset(transforms=model_transforms)
                )
            else:
                if "train_path" in dm_kwargs:
                    dset_list["train"].append(
                        dataset(dm_kwargs["train_path"], transforms=model_transforms)
                    )
                if "val_split" in dm_kwargs:
                    dset_list["val"].append(
                        dataset(dm_kwargs["val_split"], transforms=model_transforms)
                    )
                if "test_split" in dm_kwargs:
                    dset_list["test"].append(
                        dataset(dm_kwargs["test_split"], transforms=model_transforms)
                    )
        dm = MultiDataModule(
            train_dataset=MultiDataset(dset_list["train"]),
            val_dataset=MultiDataset(dset_list["val"]),
            test_dataset=MultiDataset(dset_list["test"]),
            batch_size=dm_kwargs["batch_size"],
            num_workers=dm_kwargs["num_workers"],
        )
    return dm
