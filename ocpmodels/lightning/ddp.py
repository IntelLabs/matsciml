# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: MIT License

from datetime import timedelta
import os
from typing import Callable, Optional, Any, List

from lightning_lite.plugins import CheckpointIO

import torch
from lightning_lite.plugins.collectives.torch_collective import default_pg_timeout
# majority of these imports are just for type hinting!
from pytorch_lightning.plugins.environments import (
    LightningEnvironment,
)
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.strategies import StrategyRegistry
from pytorch_lightning.plugins.precision import PrecisionPlugin


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
        accelerator: Optional["pl.accelerators.Accelerator"] = None,
        parallel_devices: Optional[List[torch.device]] = None,
        checkpoint_io: Optional[CheckpointIO] = None,
        precision_plugin: Optional[PrecisionPlugin] = None,
        ddp_comm_state: Optional[object] = None,
        ddp_comm_hook: Optional[Callable] = None,
        ddp_comm_wrapper: Optional[Callable] = None,
        model_averaging_period: Optional[int] = None,
        process_group_backend: Optional[str] = None,
        timeout: Optional[timedelta] = default_pg_timeout,
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
    find_unused_parameters=False
)

StrategyRegistry.register(
    "ddp_with_ccl",
    MPIDDPStrategy,
    description="Run distributed data parallel with an CCL environment.",
    process_group_backend="ccl",
    find_unused_parameters=False
)
