# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: MIT License

import os
from random import randint

# majority of these imports are just for type hinting!
from pytorch_lightning.plugins.environments import (
    LightningEnvironment,
)

class IntelMPIEnvironment(LightningEnvironment):
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

