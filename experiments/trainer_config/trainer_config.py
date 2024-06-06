from __future__ import annotations

import os
from copy import deepcopy

import pytorch_lightning as pl
from data_config import available_data
from model_config import available_models
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from matsciml.lightning import callbacks as cb

from matsciml.models.base import (
    BinaryClassificationTask,
    CrystalSymmetryClassificationTask,
    ForceRegressionTask,
    GradFreeForceRegressionTask,
    MaceEnergyForceTask,
    MultiTaskLitModule,
    ScalarRegressionTask,
)

trainer_args = {
    "debug": {
        "accelerator": "cpu",
        "limit_train_batches": 10,
        "limit_val_batches": 10,
        "log_every_n_steps": 1,
        "max_epochs": 2,
    },
    "experiment": {
        "accelerator": "gpu",
        "strategy": "ddp_find_unused_parameters_true",
    },
    "generic": {
        "min_epochs": 15,
        "max_epochs": 100,
    },
}


def setup_callbacks(opt_target, log_path):
    es = EarlyStopping(
        patience=5,
        monitor=opt_target,
        mode="min",
        verbose=True,
        check_finite=False,
    )
    callbacks = [
        ModelCheckpoint(monitor=opt_target, save_top_k=5),
        es,
        cb.Timer(),
        cb.GradientCheckCallback(),
        cb.SAM(),
    ]
    return callbacks


def setup_logger(log_path):
    csv_logger = CSVLogger(save_dir=log_path)
    log_path.replace("/", "-")
    cg_wb_dir = "/store/nosnap/chem-ai/wb-logs/"
    sm_wb_dir = "/workspace/nosnap/matsciml/tensornet_train/wb-logs"

    if os.path.exists(cg_wb_dir):
        save_dir = cg_wb_dir
        name = log_path.replace("/", "-")[2:]
        wb_logger = WandbLogger(
            log_model="all",
            name=name,
            save_dir=save_dir,
            project="debug",
            mode="online",
        )
    elif os.path.exists(sm_wb_dir):
        save_dir = sm_wb_dir
        name = log_path.replace("/", "-")[2:]
        wb_logger = WandbLogger(
            log_model="all",
            name=name,
            save_dir=save_dir,
            entity="smiret",
            project="mace-1M-test-sam",
            mode="online",
        )
    else:
        save_dir = "./experiments-2024-logs/wandb"
        name = log_path.replace("/", "-")[2:]
        wb_logger = WandbLogger(
            log_model="all",
            name=name,
            save_dir=save_dir,
            project="debug",
            mode="online",
        )

    return [csv_logger, wb_logger]


def setup_task(args):
    task_map = {
        "sr": ScalarRegressionTask,
        "fr": ForceRegressionTask,
        "bc": BinaryClassificationTask,
        "csc": CrystalSymmetryClassificationTask,
        "me": MaceEnergyForceTask,
        "gffr": GradFreeForceRegressionTask,
    }

    tasks = []
    if len(args.data) > 1 and len(args.tasks) <= 1:
        raise Exception("Need to specify one task per dataset")
    if len(args.data) > 1 and len(args.targets) <= 1:
        raise Exception("Need to specify one task per dataset")
    if len(args.targets) > 1 and len(args.tasks) <= 1:
        raise Exception("Need to specify one task per target")
    if len(args.targets) <= 1 and len(args.tasks) > 1:
        raise Exception("Need to specify one target per task")

    for idx, task in enumerate(args.tasks):
        task = task_map[task]
        task_args = {}
        task_args = deepcopy(available_models["generic"])
        if len(args.data) > 1:
            dset = deepcopy(available_data[args.data[idx]])
        else:
            dset = deepcopy(available_data[args.data[0]])
        normalize_kwargs = dset[args.run_type].pop("normalize_kwargs", None)
        task_loss_scaling = dset[args.run_type].pop("task_loss_scaling", None)
        task_args.update(available_models[args.model])
        if args.tasks[idx] != "csc" and args.tasks[idx] != "fr":
            task_args.update({"task_keys": [args.targets[idx]]})
        elif args.tasks[idx] == "fr":
            task_args.update({"task_keys": [args.targets[idx], "energy"]})
        task_args.update({"normalize_kwargs": normalize_kwargs})
        if task_loss_scaling is not None:
            loss_scaling = {}
            for k in task_args["task_keys"]:
                if k not in task_loss_scaling:
                    print(
                        f"\nTask key {k} does not have a loss scaling factor. Defaulting to 1.\n"
                    )
                    loss_scaling[k] = 1
                else:
                    loss_scaling[k] = task_loss_scaling[k]
            task_args.update({"task_loss_scaling": loss_scaling})
        task = task(**task_args)
        tasks.append(task)
    if len(tasks) > 1:
        datas = []
        if len(args.data) == 1:
            datas = [available_data[args.data[0]]["dataset"]] * len(tasks)
        else:
            for data in args.data:
                datas.append(available_data[data]["dataset"])

        task = MultiTaskLitModule(*list(zip(datas, tasks)))

    return task


def setup_trainer(args, callbacks, logger):
    trainer_args = deepcopy(trainer_args["generic"])
    trainer_args.update(trainer_args[args.run_type])
    num_nodes = int(args.num_nodes)
    if args.run_type == "experiment":
        trainer_args.update({"devices": args.gpus})
        trainer_args.update({"num_nodes": num_nodes})

    trainer = pl.Trainer(callbacks=callbacks, logger=logger, **trainer_args)

    return trainer
