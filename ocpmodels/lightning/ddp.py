# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: MIT License

import os
from shutil import ExecError
import sys
import __main__
from typing import Optional, Union, Dict, Any, List
import subprocess
from time import sleep

import numpy as np
import torch

# majority of these imports are just for type hinting!
from pytorch_lightning.utilities import _HYDRA_AVAILABLE
from pytorch_lightning.plugins.environments import (
    ClusterEnvironment,
    LightningEnvironment,
)
from pytorch_lightning.plugins.io.checkpoint_plugin import CheckpointIO
from pytorch_lightning.plugins.precision import PrecisionPlugin
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.strategies.launchers import _SubprocessScriptLauncher

if _HYDRA_AVAILABLE:
    from hydra.core.hydra_config import HydraConfig
    from hydra.utils import get_original_cwd, to_absolute_path


class NUMASubprocessLauncher(_SubprocessScriptLauncher):
    def __init__(
        self,
        cluster_environment: ClusterEnvironment,
        num_processes: int,
        num_nodes: int,
        numa_kwargs: Optional[List[str]] = None,
    ) -> None:
        super().__init__(cluster_environment, num_processes, num_nodes)
        self._numa_kwargs = numa_kwargs

    """
    This subprocess launcher script essentially is a copy paste of the
    original PyTorch Lightning's, and makes a small adjustment to the logic
    where `numactl` is prepended if available.
    """

    def _call_children_scripts(self) -> None:
        # bookkeeping of spawned processes
        self._check_can_spawn_children()

        # DDP Environment variables
        os.environ["MASTER_ADDR"] = self.cluster_environment.main_address
        os.environ["MASTER_PORT"] = str(self.cluster_environment.main_port)

        # allow the user to pass the node rank
        os.environ["NODE_RANK"] = str(self.cluster_environment.node_rank())
        os.environ["LOCAL_RANK"] = str(self.cluster_environment.local_rank())

        # check to see numactl is available, and if not we break informatively
        numa_check = subprocess.call(["which", "numactl"])
        if numa_check == 1:
            raise ExecError(
                "Unable to find `numactl` executable; please ensure it is available for NUMA-aware training!"
            )

        numa_call = ["numactl"]
        # the default behavior is to use local allocation, binding each process
        # to their NUMA nodes
        if not self._numa_kwargs:
            numa_call.append("-l")
        else:
            numa_call.extend(self._numa_kwargs)

        # Check if the current calling command looked like `python a/b/c.py` or `python -m a.b.c`
        # See https://docs.python.org/3/reference/import.html#main-spec
        if __main__.__spec__ is None:  # pragma: no-cover
            # Script called as `python a/b/c.py`
            # when user is using hydra find the absolute path
            path_lib = os.path.abspath if not _HYDRA_AVAILABLE else to_absolute_path

            # pull out the commands used to run the script and resolve the abs file path
            command = sys.argv
            try:
                full_path = path_lib(command[0])
            except Exception:
                full_path = os.path.abspath(command[0])

            command[0] = full_path
            # use the same python interpreter and actually running
            command = numa_call + [sys.executable] + command
        else:  # Script called as `python -m a.b.c`
            command = (
                numa_call
                + [sys.executable, "-m", __main__.__spec__.name]
                + sys.argv[1:]
            )

        os.environ["WORLD_SIZE"] = f"{self.num_processes * self.num_nodes}"

        for local_rank in range(1, self.num_processes):
            env_copy = os.environ.copy()
            env_copy["LOCAL_RANK"] = f"{local_rank}"

            # remove env var if global seed not set
            if (
                os.environ.get("PL_GLOBAL_SEED") is None
                and "PL_GLOBAL_SEED" in env_copy
            ):
                del env_copy["PL_GLOBAL_SEED"]

            # start process
            # if hydra is available and initialized, make sure to set the cwd correctly
            cwd: Optional[str] = None
            if _HYDRA_AVAILABLE:
                if HydraConfig.initialized():
                    cwd = get_original_cwd()
                    os_cwd = f'"{os.getcwd()}"'
                    command += [
                        f"hydra.run.dir={os_cwd}",
                        f"hydra.job.name=train_ddp_process_{local_rank}",
                    ]
            subprocess.Popen(command, env=env_copy, cwd=cwd)

            # starting all processes at once can cause issues
            # with dataloaders delay between 1-10 seconds
            delay = np.random.uniform(1, 5, 1)[0]
            sleep(delay)


class NUMADDPStrategy(DDPStrategy):
    """
    This DDP strategy makes a single modification to the subprocess launcher,
    calling on the modified launcher class defined above. This change
    facilitates an additional argument into `NUMADDPStrategy`, `numa_kwargs`,
    which corresponds to a list of strings that are fed into `numactl` as
    additional flags.

    By default, if `numa_kwargs` is `None`, the behavior is to have local
    processing binding, i.e. `numactl -l`.
    """

    def __init__(
        self,
        accelerator: Optional["pl.accelerators.accelerator.Accelerator"] = None,
        parallel_devices: Optional[List[torch.device]] = None,
        cluster_environment: Optional[ClusterEnvironment] = None,
        checkpoint_io: Optional[CheckpointIO] = None,
        precision_plugin: Optional[PrecisionPlugin] = None,
        ddp_comm_state: Optional[object] = None,
        ddp_comm_hook: Optional[callable] = None,
        ddp_comm_wrapper: Optional[callable] = None,
        model_averaging_period: Optional[int] = None,
        process_group_backend: Optional[str] = None,
        numa_kwargs: Optional[List[str]] = None,
        **kwargs: Union[Any, Dict[str, Any]],
    ) -> None:
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
            **kwargs,
        )
        self._numa_kwargs = numa_kwargs

    def _configure_launcher(self) -> None:
        """
        Overrides the default script launcher with a custom one that
        includes NUMA-aware `Popen` launching.
        """
        self._launcher = NUMASubprocessLauncher(
            self.cluster_environment,
            self.num_processes,
            self.num_nodes,
            self._numa_kwargs,
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
    def creates_processes_externally(self) -> bool:
        """
        Override this because we rely on `mpiexec` or `mpirun` for
        the process spawning.
        """
        return True

