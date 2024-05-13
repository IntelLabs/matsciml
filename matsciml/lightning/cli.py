# Copyright (C) 2022-4 Intel Corporation
# SPDX-License-Identifier: MIT License
from __future__ import annotations

from pytorch_lightning.cli import LightningCLI

from matsciml.models import *
from matsciml.lightning import data_utils  # noqa: F401


if __name__ == "__main__":
    cli = LightningCLI()
