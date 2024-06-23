# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: MIT License
from __future__ import annotations
from datetime import timedelta
from logging import getLogger
from typing import Union, List, Dict, Any


from matsciml.common.packages import package_registry
from pytorch_lightning.accelerators import Accelerator, AcceleratorRegistry
from pytorch_lightning.strategies import SingleDeviceStrategy, StrategyRegistry
import torch
from torch import distributed as dist

default_pg_timeout = timedelta(seconds=1800)

logger = getLogger(__file__)

if package_registry["ipex"]:
    try:
        import intel_extension_for_pytorch as ipex  # noqa: F401
    except ImportError as e:
        logger.warning(f"Unable to import IPEX due to {e} - XPU may not function.")

    __all__ = ["XPUAccelerator", "SingleXPUStrategy"]

    class XPUAccelerator(Accelerator):
        """
        Implements a Lightning Accelerator class for Intel GPU usage. Depends
        on Intel Extension for PyTorch to be installed.
        """

        @staticmethod
        def parse_devices(devices: Union[int, List[int]]) -> List[int]:
            """
            Parse the `trainer` input for devices and homogenize them.
            Parameters
            ----------
            devices : Union[int, List[int]]
                Single or list of device numbers to use
            Returns
            -------
            List[int]
                List of device numbers to use
            """
            if isinstance(devices, int):
                # assume that this is the number of devices to use
                devices = list(range(devices))
            return devices

        def setup_device(self, device: torch.device) -> None:
            """
            Configure the current process to use a specified device.
            Perhaps unreliably and misguiding, the IPEX implementation of this method
            tries to mirror the CUDA version but `ipex.xpu.set_device` actually refuses
            to accept anything other than an index. I've tried to work around this
            by grabbing the index from the device if possible, and just setting
            it to the first device if not using a distributed/multitile setup.
            """
            # first try and see if we can grab the index from the device
            index = getattr(device, "index", None)
            if index is None and not dist.is_initialized():
                index = 0
            torch.xpu.set_device(index)

        def teardown(self) -> None:
            # as it suggests, this is run on cleanup
            torch.xpu.empty_cache()

        def get_device_stats(self, device) -> Dict[str, Any]:
            return torch.xpu.memory_stats(device)

        @staticmethod
        def get_parallel_devices(devices: List[int]) -> List[torch.device]:
            """
            Return a list of torch devices corresponding to what is available.
            Essentially maps indices to `torch.device` objects.
            Parameters
            ----------
            devices : List[int]
                List of integers corresponding to device numbers
            Returns
            -------
            List[torch.device]
                List of `torch.device` objects for each device
            """
            return [torch.device("xpu", i) for i in devices]

        @staticmethod
        def auto_device_count() -> int:
            # by default, PVC has two tiles per GPU
            return torch.xpu.device_count()

        @staticmethod
        def is_available() -> bool:
            """
            Determines if an XPU is actually available.
            Returns
            -------
            bool
                True if devices are detected, otherwise False
            """
            try:
                return torch.xpu.device_count() != 0
            except (AttributeError, NameError):
                return False

        @classmethod
        def register_accelerators(cls, accelerator_registry) -> None:
            accelerator_registry.register(
                "xpu",
                cls,
                description="Intel Data Center GPU Max - codename Ponte Vecchio",
            )

    # add PVC to the registry
    AcceleratorRegistry.register("xpu", XPUAccelerator)

    class SingleXPUStrategy(SingleDeviceStrategy):
        """
        This class implements the strategy for using a single PVC tile.
        """

        strategy_name = "pvc_single"

        def __init__(
            self,
            device: str | None = "xpu",
            checkpoint_io=None,
            precision_plugin=None,
        ):
            super().__init__(
                device=device,
                accelerator=XPUAccelerator(),
                checkpoint_io=checkpoint_io,
                precision_plugin=precision_plugin,
            )

        @property
        def is_distributed(self) -> bool:
            return False

        def setup(self, trainer) -> None:
            self.model_to_device()
            super().setup(trainer)

        def setup_optimizers(self, trainer) -> None:
            super().setup_optimizers(trainer)

        def model_to_device(self) -> None:
            self.model.to(self.root_device)

        @classmethod
        def register_strategies(cls, strategy_registry) -> None:
            strategy_registry.register(
                cls.strategy_name,
                cls,
                description=f"{cls.__class__.__name__} - uses a single XPU tile for compute.",
            )

    StrategyRegistry.register(
        "single_xpu",
        SingleXPUStrategy,
        description="Strategy utilizing a single Intel GPU device or tile.",
    )
