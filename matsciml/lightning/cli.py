# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: MIT License
from __future__ import annotations

import pytorch_lightning as pl
from pytorch_lightning.utilities.cli import (
    DATAMODULE_REGISTRY,
    MODEL_REGISTRY,
    LightningCLI,
)

from matsciml import models
from matsciml.lightning import data_utils

"""
This module interfaces with the PyTorch Lightning CLI, and when called, allows
the user to define YAML configuration files for reproducible and modular
training, testing, and development.

All that is really done in this module is inform the PyTorch Lightning registry
where to look for models (`MODEL_REGISTRY`) and data (`DATAMODULE_REGISTRY`).
The former is set up to look for children of `LightningModule`, which comprises
the task `LightningModule`s, and models like DimeNetPP that are implemented with
`AbstractTask`, which in turn also inherits from `LightningModule` (this might change later).

To check what tasks and data modules have been successfully registered, import this
module, and print `MODEL_REGISTRY` and/or `DATAMODULE_REGISTRY`: if your model was
included in the namespace correctly, it should appear there.

To use the CLI, all one needs to do is write a YAML configuration file, and then
run `python -m matsciml.lightning.cli fit --config <CONFIG>.yml`, substituting
fit with any other appropriate task, and <CONFIG> with the name of your configuration
file.
"""


# this registers the task classes implemented as LightningModules
MODEL_REGISTRY.register_classes(models, pl.LightningModule)

# this registers the data
DATAMODULE_REGISTRY.register_classes(data_utils, pl.LightningDataModule)


if __name__ == "__main__":
    cli = LightningCLI()
