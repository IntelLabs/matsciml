from __future__ import annotations

from abc import abstractmethod, ABC
from typing import Literal, Generator

import numpy as np
from pytorch_lightning import Trainer, LightningModule


class BaseScalingSchedule(ABC):
    """
    Implements the abstract class for scaling schedulers.

    Subclasses will implement the actual schedules, and all of
    the boilerplate (e.g. setup and step methods) are lifted here.
    """

    @property
    def grid(self) -> np.ndarray:
        return self._grid

    @abstractmethod
    def set_grid(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "Requires concrete method of computing time grid for scheduling."
        )

    def __len__(self) -> int:
        return len(self.grid)

    @property
    def step_frequency(self) -> Literal["step", "epoch"]:
        return self._step_frequency

    @step_frequency.setter
    def step_frequency(self, value: Literal["step", "epoch"]) -> None:
        assert value in [
            "step",
            "epoch",
        ], "Invalid step frequency; only step/epoch are supported."
        self._step_frequency = value

    def step(self) -> float:
        """Function that returns a value at some point in time."""
        return next(self.schedule)

    @property
    @abstractmethod
    def schedule(self) -> Generator[float, None, None]:
        raise NotImplementedError(
            "Must override `schedule` property and setter methods."
        )

    @abstractmethod
    def setup(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Configures the schedule by grabbing whatever is needed from trainer/module"""
        ...
