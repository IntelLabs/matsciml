# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: MIT License
from __future__ import annotations

import os
import socket
from datetime import timedelta
from typing import Any, Callable
from contextlib import nullcontext

import torch
from torch import distributed as dist
from torch import nn
from torch.nn.parallel.distributed import DistributedDataParallel
import pytorch_lightning as pl
from pytorch_lightning.plugins import CheckpointIO
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.plugins.environments import LightningEnvironment
from pytorch_lightning.plugins.precision import Precision
from pytorch_lightning.strategies import StrategyRegistry

from matsciml.common.packages import package_registry


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

    def __init__(self, main_address: str | None = None, main_port: int | None = None):
        self.main_address = main_address
        self.main_port = main_port

    def world_size(self) -> int:
        return int(os.environ["PMI_SIZE"])

    def local_rank(self) -> int:
        return int(os.environ["MPI_LOCALRANKID"])

    def global_rank(self) -> int:
        return int(os.environ["PMI_RANK"])

    @property
    def main_address(self) -> str:
        return self._main_address

    @main_address.setter
    def main_address(self, value: str | None):
        if not value:
            value = os.getenv("HYDRA_BSTRAP_LOCALHOST", None)
        if not value:
            raise ValueError(
                "No main address passed, and MPI did not set HYDRA_BSTRAP_LOCALHOST."
            )
        self._main_address = value
        os.environ["MASTER_ADDR"] = self._main_address

    @property
    def main_port(self) -> int:
        return self._main_port

    @main_port.setter
    def main_port(self, value: int | None):
        if not value:
            value = 30256
        # check to make sure port and address are accessible
        self._main_port = value
        os.environ["MASTER_PORT"] = str(self._main_port)

    @staticmethod
    def _validate_address_port(addr: str, port: int) -> bool:
        obj = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = obj.connect_ex((addr, port)) == 0
        obj.close()
        return result

    @property
    def creates_processes_externally(self) -> bool:
        """
        Override this because we rely on `mpiexec` or `mpirun` for
        the process spawning.
        """
        return True

    @property
    def local_world_size(self) -> int:
        """Return the number of devices per node."""
        return int(os.environ["MPI_LOCALNRANKS"])

    @property
    def num_nodes(self) -> int:
        """Return the of numbers, based on ranks per node and global world size."""
        num_nodes = self.world_size() // self.local_world_size
        return num_nodes


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
        cluster_environment: MPIEnvironment | None = None,
        **kwargs: Any,
    ) -> None:
        if not cluster_environment:
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
        if not dist.is_initialized():
            dist.init_process_group(
                self.process_group_backend,
                init_method=f"tcp://{addr}:{port}",
                world_size=self.cluster_environment.world_size(),
                rank=self.cluster_environment.global_rank(),
            )
        # this is to force initialization of distributed backend
        dummy = torch.ones((5, 2), device=self.root_device)
        dist.broadcast(dummy, src=0)

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

if package_registry["ipex"] and hasattr(torch, "xpu"):
    StrategyRegistry.register(
        "ddp_with_xpu",
        MPIDDPStrategy,
        description="Run distributed data parallel on Intel XPUs.",
        process_group_backend="ccl",
        accelerator="xpu",
    )
