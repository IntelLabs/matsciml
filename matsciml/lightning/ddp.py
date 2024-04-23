# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: MIT License
from __future__ import annotations

import os
from datetime import timedelta
from typing import Any, Callable
from contextlib import nullcontext

import torch
from torch import distributed as dist
from torch import nn
from torch.nn.parallel.distributed import DistributedDataParallel
import pytorch_lightning as pl
from pytorch_lightning.plugins import CheckpointIO
from pytorch_lightning.plugins.environments import LightningEnvironment
from pytorch_lightning.plugins.environments.lightning import find_free_network_port
from pytorch_lightning.plugins.precision import Precision
from pytorch_lightning.strategies import StrategyRegistry
from pytorch_lightning.strategies.ddp import DDPStrategy


__all__ = ["MPIEnvironment", "MPIDDPStrategy"]

default_pg_timeout = timedelta(seconds=1800)


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
        # find an open port
        port = int(os.getenv("MASTER_PORT", find_free_network_port()))
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
        precision_plugin: Precision | None = None,
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

    def setup_distributed(self):
        """Overrides base method so we can perform dummy all_reduce."""
        port = self.cluster_environment.main_port
        addr = self.cluster_environment.main_address
        dist.init_process_group(
            self.process_group_backend,
            init_method=f"tcp://{addr}:{port}",
            world_size=self.cluster_environment.world_size(),
            rank=self.cluster_environment.global_rank(),
        )
        # this is to force initialization of distributed backend
        dummy = torch.ones((5, 2), device=self.root_devce)
        dist.all_reduce(dummy)

    def _setup_model(self, model: nn.Module) -> DistributedDataParallel:
        device_ids = self.determine_ddp_device_ids()
        # this enforces an XPU stream, instead of CUDA
        if device_ids is not None and hasattr(torch, "xpu"):
            ctx = torch.xpu.StreamContext(torch.xpu.current_stream())
        else:
            ctx = nullcontext()
        with ctx:
            return DistributedDataParallel(
                module=model, device_ids=device_ids, **self._ddp_kwargs
            )

    def teardown(self):
        """Ensure that distributed processes close gracefully."""
        super().teardown()
        if dist.is_initialized():
            dist.destroy_process_group()


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

StrategyRegistry.register(
    "ddp_with_xpu",
    MPIDDPStrategy,
    description="Run distributed data parallel on Intel XPUs.",
    process_group_backend="ccl",
    accelerator="xpu",
)
