from __future__ import annotations

from abc import abstractmethod, ABC
from typing import Literal, Generator
from functools import cached_property
from logging import getLogger

import numpy as np
from pytorch_lightning import Trainer, LightningModule

__all__ = ["LinearScalingSchedule", "SigmoidScalingSchedule"]


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
        if (step_count := trainer.max_steps) > -1:
            self.set_grid(step_count)
        else:
            train_loader = trainer.datamodule.train_dataloader()
            # num_train_batches is how many batches loaded up per loader, per epoch
            expected_epochs = trainer.max_epochs
            if self.step_frequency == "step":
                if trainer.limit_train_batches <= 1.0:
                    num_train_batches = len(train_loader)
                else:
                    num_train_batches = trainer.limit_train_batches
                num_steps = int(num_train_batches * expected_epochs)
            else:
                num_steps = expected_epochs
            self.set_grid(num_steps)
        # set the initial scaling value
        pl_module.task_loss_scaling[self.key] = self.initial_value

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} for {self.key} - start: {self.initial_value}, end: {self.end_value}"


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


class SigmoidScalingSchedule(BaseScalingSchedule):
    def __init__(
        self,
        key: str,
        initial_value: float,
        end_value: float,
        center_frac: float,
        curvature: float = 5e-7,
        step_frequency: Literal["step", "epoch"] = "step",
    ) -> None:
        """
        Schedules a sigmoidal scheduling curve.

        This provides an exponential ramp between initial and
        end values, with controllable turning point and rate
        of change.

        Parameters
        ----------
        key : str
            Task key that this schedule maps to.
        initial : float
            Value of the left asymptote; i.e. for t < 0.
        end : float
            Value of the right asymptote; i.e. for t > 0.
        center_frac : float
            For t in [0,1], this is the turning point of the
            curve. Assuming this range spans the whole training
            run, 0.5 would place the turning point at half the
            epochs/training steps.
        curvature : float
            The rate at which the values change. Small values of `curvature`
            (in orders of magnitude) increases the rate of change such that
            the changeover becomes much more abrupt, spending more steps/epochs
            closer to their asymptotes. Values between 1e-2 and 1e-9 are
            recommended; larger values will have a very gradual ramp up but
            will not necessarily obey the initial value exactly.
        step_frequency : Literal['step', 'epoch']
            Frequency at which this schedule works. This dictates
            the number of grid points we expect to ramp with.
        """
        super().__init__()
        self.key = key
        self.initial_value = initial_value
        self.end_value = end_value
        self.step_frequency = step_frequency
        assert 0.0 < center_frac < 1.0, "Center fraction value must be between [0,1]"
        self.center_frac = center_frac
        self.curvature = curvature
        self._logger = getLogger(f"matsciml.sigmoid_loss_scaling.{key}")
        if curvature > 1e-2:
            self._logger.warning(
                "Curvature value is larger than expected; make sure this is intended!"
            )
        if curvature < 1e-9:
            self._logger.warning(
                "Curvature value is very small; make sure this is intended!"
            )

    @staticmethod
    def sigmoid_curve(
        t: float | np.ndarray,
        initial: float,
        end: float,
        center_frac: float,
        curvature: float,
    ) -> float | np.ndarray:
        """
        Returns the value at point t for a parametrized sigmoid curve.

        This function is plotted at this link: https://www.desmos.com/calculator/urrjjviigq?lang=en
        Essentially, it ramps between `initial` and `end` values, with
        the turning point centered at `center_frac`, with the rate of
        change determined by curvature.

        The explicit assumption  here is that all of the 'action' occurs
        between t in [0,1], although it is defined beyond those ranges
        (i.e. the initial and end values).

        Parameters
        ----------
        t : float
            Equivalent to `x`, acting as the ordinate to this function.
            While the function is defined for all `t`, this function
            mainly assumes t is between [0,1].
        initial : float
            Value of the left asymptote; i.e. for t < 0.
        end : float
            Value of the right asymptote; i.e. for t > 0.
        center_frac : float
            For t in [0,1], this is the turning point of the
            curve. Assuming this range spans the whole training
            run, 0.5 would place the turning point at half the
            epochs/training steps.
        curvature : float
            The rate at which the values change. Small values of `curvature`
            (in orders of magnitude) increases the rate of change such that
            the changeover becomes much more abrupt, spending more steps/epochs
            closer to their asymptotes.

        Returns
        -------
        float
            Value of the curve at point t.
        """
        k_c = curvature**center_frac
        k_t = curvature**t
        numerator = end * k_c + initial * k_t
        denominator = k_c + k_t
        return numerator / denominator

    @cached_property
    def schedule(self) -> Generator[float, None, None]:
        grid = self.grid
        schedule = self.sigmoid_curve(
            grid, self.initial_value, self.end_value, self.center_frac, self.curvature
        )
        for value in schedule:
            yield value

    def set_grid(self, total_steps: int, *args, **kwargs) -> None:
        self._grid = np.linspace(0.0, 1.0, total_steps)
