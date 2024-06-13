from __future__ import annotations

from copy import deepcopy

import pytorch_lightning as pl

from experiments.utils.utils import instantiate_arg_dict


def setup_trainer(input_args, experiment_type):
    trainer_args = instantiate_arg_dict(deepcopy(input_args))
    if "loggers" in trainer_args:
        loggers = []
        for logger in trainer_args["loggers"]:
            loggers.append(logger)
        trainer_args.pop("loggers")
    if "callbacks" in trainer_args:
        callbacks = []
        for callback in trainer_args["callbacks"]:
            callbacks.append(callback)
        trainer_args.pop("callbacks")

    trainer_kwargs = input_args["generic"]
    trainer_kwargs.update(input_args[experiment_type])
    trainer = pl.Trainer(logger=loggers, callbacks=callbacks, **trainer_kwargs)
    return trainer
