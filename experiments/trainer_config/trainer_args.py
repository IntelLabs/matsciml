from pytorch_lightning.loggers import CSVLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from matsciml.lightning import callbacks as cb

from training_script import opt_target, log_path


trainer_args = {
    "callbacks": [
        {
            "callback": EarlyStopping,
            "args": {
                "patience": 5,
                "monitor": opt_target,
                "mode": "min",
                "verbose": True,
                "check_finite": False,
            },
        },
        {
            "callback": ModelCheckpoint,
            "args": {
                "monitor": opt_target,
                "save_top_k": 3,
            },
        },
        {
            "callback": cb.Timer,
        },
        {"callback": cb.GradientCheckCallback},
        {"callback": cb.SAM},
    ],
    "loggers": [
        {
            "logger": CSVLogger,
            "args": {"save_dir": log_path},
        },
        {
            "logger": WandbLogger,
            "args": {
                "log_mode": "all",
                "name": name,
                "project": "debug",
                "mode": "online",
            },
        },
    ],
}


def setup_trainer_args():
    pass
