from __future__ import annotations

import pytest
import tempfile

from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import EarlyStopping

from experiments.trainer_config import setup_trainer


@pytest.fixture
def trainer_args() -> dict:
    args = {
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
        "generic": {"min_epochs": 15, "max_epochs": 100},
        "callbacks": [
            {
                "class_path": "lightning.pytorch.callbacks.EarlyStopping",
                "init_args": [
                    {"patience": 5},
                    {"mode": "min"},
                    {"verbose": True},
                    {"check_finite": False},
                    {"monitor": "val_energy"},
                ],
            },
            {
                "class_path": "lightning.pytorch.callbacks.ModelCheckpoint",
                "init_args": [{"save_top_k": 3}, {"monitor": "val_energy"}],
            },
        ],
        "loggers": [
            {
                "class_path": "lightning.pytorch.loggers.CSVLogger",
                "init_args": {"save_dir": "./temp"},
            },
        ],
    }
    return args


@pytest.mark.dependency(depends=["trainer_args"])
def test_trainer_setup(trainer_args):
    temp_dir = tempfile.TemporaryDirectory()
    trainer = setup_trainer(
        {"run_type": "debug", "log_path": f"{temp_dir}", "cli_args": None}, trainer_args
    )
    assert any([CSVLogger == logger.__class__ for logger in trainer.loggers])
    assert any([EarlyStopping == logger.__class__ for logger in trainer.callbacks])
    assert trainer.max_epochs == 2
    temp_dir.cleanup()
