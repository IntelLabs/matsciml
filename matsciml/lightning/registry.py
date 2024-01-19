# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: MIT License
from __future__ import annotations

import pytorch_lightning as pl
from pytorch_lightning.utilities.cli import DATAMODULE_REGISTRY, MODEL_REGISTRY
from torch import nn

from matsciml import models
from matsciml.lightning import data_utils

# add more models here; either export them into the `models` namespace,
# or register specific modules (e.g. `models.schnet`). The registration
# will look for children of the class you specify, e.g. `nn.Module`, or
# `pl.LightningModule`
MODEL_REGISTRY.register_classes(models.base, pl.LightningModule)

DATAMODULE_REGISTRY.register_classes(data_utils, pl.LightningDataModule)
