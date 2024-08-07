from __future__ import annotations

from abc import abstractmethod, ABC
from typing import Literal, Generator
from functools import cached_property

import numpy as np
from pytorch_lightning import Trainer, LightningModule

__all__ = ["LinearScalingSchedule"]


class BaseScalingSchedule(ABC):
    """
    Implements the abstract class for scaling schedulers.

    Subclasses will implement the actual schedules, and all of
    the boilerplate (e.g. setup and step methods) are lifted here.
    """

    @property
    def key(self) -> str:
        """References the target key this scaling value maps to."""
        return self._key

    @key.setter
    def key(self, value: str) -> None:
        self._key = value

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


class LinearScalingSchedule(BaseScalingSchedule):
    def __init__(
        self,
        key: str,
        initial_value: float,
        end_value: float | None = None,
        step_frequency: Literal["step", "epoch"] = "epoch",
    ) -> None:
        super().__init__()
        self.key = key
        self.initial_value = initial_value
        if not end_value:
            end_value = initial_value
        self.end_value = end_value
        self.step_frequency = step_frequency

    @cached_property
    def schedule(self) -> Generator[float, None, None]:
        delta = np.abs(self.initial_value - self.end_value)
        delta_values = self.grid * delta
        if self.initial_value > self.end_value:
            delta_values = np.negative(delta_values)
        # linear ramp to go from initial to end values
        schedule = delta_values + self.initial_value
        for value in schedule:
            yield value

    def set_grid(self, total_steps: int, *args, **kwargs) -> None:
        self._grid = np.linspace(0.0, 1.0, total_steps)

    def setup(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """
        This configures the grid based on either the total number of
        projected steps or epochs, depending on what was specified.

        Parameters
        ----------
        trainer : Trainer
            Instance of a PyTorch Lightning trainer.
        pl_module : LightningModule
            Instances of a LightningModule; not used in this setup.
        """
        if step_count := trainer.max_steps:
            self.set_grid(step_count)
        else:
            train_loader = trainer.train_dataloader()
            # num_train_batches is how many batches loaded up per loader, per epoch
            expected_epochs = trainer.max_epochs
            if self.step_frequency == "step":
                num_train_batches = len(train_loader)
                num_steps = int(num_train_batches * expected_epochs)
            else:
                num_steps = expected_epochs
            self.set_grid(num_steps)
