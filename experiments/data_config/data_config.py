from __future__ import annotations

import json
import os
import sys
from copy import deepcopy

from matsciml.datasets import *

from matsciml.lightning.data_utils import (
    MatSciMLDataModule,
    MultiDataModule,
    MultiDataset,
)

from experiments.data_config import available_data

data_keys = list(available_data.keys())

norm_files = os.listdir("./experiments/norms")
norm_dict = {}
for data_name in data_keys:
    norm_dict[data_name] = None
    for file in norm_files:
        if data_name in file:
            norm_dict[data_name] = json.load(
                open(os.path.join("./experiments/norms", file))
            )


available_data = {
    "generic": {"experiment": {"batch_size": 4, "num_workers": 16}},
}


def setup_datamodule(args):
    if len(args.data) == 1:
        data = args.data[0]
        dset = deepcopy(available_data[data])
        dm_kwargs = deepcopy(available_data["generic"]["experiment"])
        dset[args.run_type].pop("normalize_kwargs", None)
        dset[args.run_type].pop("task_loss_scaling", None)
        dm_kwargs.update(dset[args.run_type])
        if args.run_type == "debug":
            dm = MatSciMLDataModule.from_devset(
                dataset=dset["dataset"],
                dset_kwargs={"transforms": transforms[args.model]},
                **dm_kwargs,
            )
        else:
            dm = MatSciMLDataModule(
                dataset=dset["dataset"],
                dset_kwargs={"transforms": transforms[args.model]},
                **dm_kwargs,
            )
    else:
        train_dset_list = []
        val_dset_list = []
        for data in args.data:
            dset = deepcopy(available_data[data])
            dm_kwargs = deepcopy(available_data["generic"]["experiment"])
            dset[args.run_type].pop("normalize_kwargs", None)
            dm_kwargs.update(dset[args.run_type])
            dataset_name = dset["dataset"]
            dataset = getattr(sys.modules[__name__], dataset_name)
            model_transforms = transforms[args.model]
            train_dset_list.append(
                dataset(dm_kwargs["train_path"], transforms=model_transforms)
            )
            val_dset_list.append(
                dataset(dm_kwargs["val_split"], transforms=model_transforms)
            )

        train_dset = MultiDataset(train_dset_list)
        val_dset = MultiDataset(val_dset_list)
        dm = MultiDataModule(
            train_dataset=train_dset,
            val_dataset=val_dset,
            batch_size=dm_kwargs["batch_size"],
            num_workers=dm_kwargs["num_workers"],
        )
    return dm
