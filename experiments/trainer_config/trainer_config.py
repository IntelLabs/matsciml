from __future__ import annotations

from copy import deepcopy
from typing import Any

import lightning.pytorch as pl

from experiments.utils.utils import instantiate_arg_dict, update_arg_dict


def setup_extra_trainer_args(
    log_path: str, trainer_args: dict[str, Any]
) -> dict[str, Any]:
    if "loggers" in trainer_args:
        for logger in trainer_args["loggers"]:
            if "CSVLogger" in logger["class_path"]:
                logger.setdefault("init_args", {})
                if "save_dir" not in logger["init_args"]:
                    logger["init_args"].update({"save_dir": log_path})
            if "WandbLogger" in logger["class_path"]:
                logger.setdefault("init_args", {})
                if "name" not in logger["init_args"]:
                    logger["init_args"].update({"name": log_path})
    return trainer_args


def setup_trainer(config: dict[str, Any], trainer_args: dict[str, Any]) -> pl.Trainer:
    run_type = config["run_type"]
    trainer_args = setup_extra_trainer_args(config["log_path"], trainer_args)
    trainer_args = instantiate_arg_dict(deepcopy(trainer_args))
    trainer_args = update_arg_dict("trainer", trainer_args, config["cli_args"])
    # if loggers were requested, configure them
    if "loggers" in trainer_args:
        loggers = []
        for logger in trainer_args["loggers"]:
            loggers.append(logger)
        trainer_args.pop("loggers")
    else:
        loggers = None
    # if callbacks were requested, configure them
    if "callbacks" in trainer_args:
        callbacks = []
        for callback in trainer_args["callbacks"]:
            callbacks.append(callback)
        trainer_args.pop("callbacks")
    else:
        callbacks = None

    trainer_kwargs = trainer_args["generic"]
    trainer_kwargs.update(trainer_args[run_type])
    trainer = pl.Trainer(logger=loggers, callbacks=callbacks, **trainer_kwargs)
    return trainer
