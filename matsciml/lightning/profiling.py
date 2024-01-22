# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: MIT License
from __future__ import annotations

from pathlib import Path
from subprocess import run
from typing import Union

import psutil
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from torch import distributed as dist


def is_vtune_running() -> bool:
    """
    Simple utility function that tells you if a VTune process is underway.

    Returns
    -------
    bool
        True if a process named "vtune" exists in the process list.
    """
    return "vtune" in (p.name() for p in psutil.process_iter())


class VTuneResume(Callback):
    """
    This Callback will automatically resume VTune collection as the training loop
    begins. The idea behind this is to help VTune focus specifically on computation
    data collection, rather than also capturing all of the overhead associated with
    launching the training loop.

    The VTune resume call will only execute on rank zero

    The usage would be:

    ```
    resume_callback = VTuneResume("vtune_output")
    trainer = pl.Trainer(callbacks=[resume_callback])
    ```
    """

    def __init__(self, result_dir: str | Path) -> None:
        super().__init__()
        # result dir is necessary for VTune resume command to work
        if isinstance(result_dir, str):
            result_dir = Path(result_dir)
        self._result_dir = str(result_dir)
        self._vtune_is_running = is_vtune_running()
        self._has_resumed = False

    def _resume_vtune(self) -> None:
        # this logic will execute if we are not using multiple workers, or if
        # we're on rank zero
        if any([not dist.is_initialized(), dist.get_rank() == 0]):
            if self._vtune_is_running and not self._has_resumed:
                _ = run(["vtune", "-command", "resume", "-r", self._result_dir])
                # set the state so we don't run it every training loop
                self._has_resumed = True

    def _stop_vtune(self) -> None:
        # this logic implements the opposite of start: we will stop data collection
        # once the loop has ended.
        if any([not dist.is_initialized(), dist.get_rank() == 0]):
            if self._vtune_is_running and self._has_resumed:
                _ = run(["vtune", "-command", "stop", "-r", self._result_dir])

    def on_validation_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        self._resume_vtune()

    def on_validation_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        self._stop_vtune()

    def on_train_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        self._resume_vtune()

    def on_train_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        pass
