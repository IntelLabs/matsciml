# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: MIT License
from __future__ import annotations

import os
from datetime import timedelta
from typing import Any, Callable, List, Optional

import torch
from lightning_fabric.plugins.collectives.torch_collective import default_pg_timeout
from pytorch_lightning.plugins import CheckpointIO
from pytorch_lightning.plugins.environments import LightningEnvironment
from pytorch_lightning.plugins.precision import PrecisionPlugin
from pytorch_lightning.strategies import StrategyRegistry
from pytorch_lightning.strategies.ddp import DDPStrategy

# majority of these imports are just for type hinting!


class MPIEnvironment(LightningEnvironment):
    """
    This environment specializes in the use of Intel MPI for distributed
    multiworker instances. The key assumptions for using this environment
    are:

    1. The use of Intel MPI
    2. The launch script utilizes PyTorch Lightning abstractions
    3. The launch script is used via `mpiexec -n -ppn ... python train.py

    The main motivation behind this environment is two-fold: to keep the
    `pl.Trainer` functionality, while maintaining the ability to work with
    NUMA bindings (e.g. via `-map-by numa`) to ensure optimal CPU/memory
    utilization.
    """

    def world_size(self) -> int:
        return int(os.environ["PMI_SIZE"])

    def local_rank(self) -> int:
        return int(os.environ["MPI_LOCALRANKID"])

    def global_rank(self) -> int:
        return int(os.environ["PMI_RANK"])

    @property
    def main_address(self) -> str:
        return os.environ["HYDRA_BSTRAP_LOCALHOST"]

    @property
    def main_port(self) -> int:
        port = int(os.getenv("MASTER_PORT", "12345"))
        return port

    @property
    def creates_processes_externally(self) -> bool:
        """
        Override this because we rely on `mpiexec` or `mpirun` for
        the process spawning.
        """
        return True


class MPIDDPStrategy(DDPStrategy):
    def __init__(
        self,
        accelerator: pl.accelerators.Accelerator | None = None,
        parallel_devices: list[torch.device] | None = None,
        checkpoint_io: CheckpointIO | None = None,
        precision_plugin: PrecisionPlugin | None = None,
        ddp_comm_state: object | None = None,
        ddp_comm_hook: Callable | None = None,
        ddp_comm_wrapper: Callable | None = None,
        model_averaging_period: int | None = None,
        process_group_backend: str | None = None,
        timeout: timedelta | None = default_pg_timeout,
        **kwargs: Any,
    ) -> None:
        cluster_environment = MPIEnvironment()
        if process_group_backend:
            assert process_group_backend in [
                "ccl",
                "mpi",
            ], f"Unsupported distributed backend! {process_group_backend}"
        super().__init__(
            accelerator,
            parallel_devices,
            cluster_environment,
            checkpoint_io,
            precision_plugin,
            ddp_comm_state,
            ddp_comm_hook,
            ddp_comm_wrapper,
            model_averaging_period,
            process_group_backend,
            timeout,
            **kwargs,
        )


StrategyRegistry.register(
    "ddp_with_mpi",
    MPIDDPStrategy,
    description="Run distributed data parallel with an MPI environment.",
    process_group_backend="mpi",
)

StrategyRegistry.register(
    "ddp_with_ccl",
    MPIDDPStrategy,
    description="Run distributed data parallel with an CCL environment.",
    process_group_backend="ccl",
)
