from __future__ import annotations
from pathlib import Path

from typing import Callable, Literal

import torch
from ase import Atoms
from ase.calculators.calculator import Calculator
import numpy as np

from matsciml.common.types import DataDict
from matsciml.models.base import (
    ScalarRegressionTask,
    GradFreeForceRegressionTask,
    ForceRegressionTask,
    MultiTaskLitModule,
)
from matsciml.datasets.transforms.base import AbstractDataTransform

__all__ = ["MatSciMLCalculator"]


def recursive_type_cast(
    data_dict: DataDict,
    dtype: torch.dtype,
    ignore_keys: list[str] = ["atomic_numbers"],
    convert_numpy: bool = True,
) -> DataDict:
    """
    Recursively cast a dictionary of data into a particular
    numeric type.

    This function will only type cast torch tensors; the ``convert_numpy``
    argument will optionally convert NumPy arrays into tensors first,
    _then_ perform the type casting.

    Parameters
    ----------
    data_dict : DataDict
        Dictionary of data to recurse through.
    dtype : torch.dtype
        Data type to convert to.
    ignore_keys : list[str]
        Keys to ignore in the process; useful for excluding
        casting for certain things like ``atomic_numbers``
        that are intended to be ``torch.long`` from being
        erroneously casted to floats.
    convery_numpy : bool, default True
        If True, converts NumPy arrays into PyTorch tensors
        before performing type casting.

    Returns
    -------
    DataDict
        Data dictionary with type casted results.
    """
    for key, value in data_dict.items():
        if ignore_keys and key in ignore_keys:
            continue
        # optionally convert numpy arrays into torch tensors
        # prior to type casting
        if isinstance(value, np.ndarray) and convert_numpy:
            value = torch.from_numpy(value)
        if isinstance(value, dict):
            data_dict[key] = recursive_type_cast(value, dtype)
        if isinstance(value, torch.Tensor):
            data_dict[key] = value.to(dtype)
    return data_dict


def __checkpoint_conversion_exist_check(ckpt_path: str | Path) -> Path:
    """Standardizes and checks for checkpoint path existence."""
    if isinstance(ckpt_path, str):
        ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found; passed {ckpt_path}")
    return ckpt_path


class MatSciMLCalculator(Calculator):
    implemented_properties = ["energy", "forces", "stress", "dipole"]

    def __init__(
        self,
        task_module: ScalarRegressionTask
        | GradFreeForceRegressionTask
        | ForceRegressionTask
        | MultiTaskLitModule,
        transforms: list[AbstractDataTransform | Callable] | None = None,
        restart=None,
        label=None,
        atoms: Atoms | None = None,
        directory=".",
        conversion_factor: float | dict[str, float] = 1.0,
        **kwargs,
    ):
        """
        Initialize an instance of the ``MatSciMLCalculator`` used by ``ase``
        simulations.

        This class essentially acts as an adaptor to a select number of
        ``matsciml`` tasks by converting ``Atoms`` data structures into
        those expected by ``matsciml`` models, and then extracting the
        output of the forward pass into the expected results dictionary
        for ``ase``.

        The recommended mode of usage of this class is to use one of the
        constructor methods, e.g. ``MatSciMLCalculator.from_pretrained_force_regression``,
        to set up the calculator based on one of the supported tasks.
        A list of transforms can be passed as well in order to reuse the
        same transformation pipeline as the rest of ``matsciml``.

        Examples
        ---------
        Create from a pretrained ``ForceRegressionTask``

        >>> calc = MatSciMLCalculator.from_pretrained_force_regression(
            "lightning_logs/version_10/checkpoints/epoch=10-step=3000.ckpt",
            transforms=[PeriodicPropertiesTransform(6.0), PointCloudToGraphTransform("pyg")]
        )

        Parameters
        ----------
        task_module
            Instance of a supported ``matsciml`` task. What is 'supported' is
            intended to reflect the kinds of modeling tasks, e.g. energy/force
            prediction.
        transforms : list[AbstractDataTransform | Callable] | None, default None
            An optional list of transforms, similar to what is used in the rest
            of the ``matsciml`` pipeline.
        restart
            Argument passed into ``ase`` Calculator base class.
        label
            Argument passed into ``ase`` Calculator base class.
        atoms : Atoms | None, default None
            Optional ``Atoms`` object to attach this calculator to.
        directory
            Argument passed into ``ase`` Calculator base class.
        conversion_factor : float | dict[str, float]
            Conversion factors to each property, specified as key/value
            pairs where keys refer to data in ``self.results`` reported
            to ``ase``. If a single ``float`` is passed, we assume that
            the conversion is applied to the energy output. Each factor
            is multiplied with the result.
        """
        super().__init__(
            restart, label=label, atoms=atoms, directory=directory, **kwargs
        )
        assert isinstance(
            task_module,
            (
                ForceRegressionTask,
                ScalarRegressionTask,
                GradFreeForceRegressionTask,
                MultiTaskLitModule,
            ),
        ), f"Expected task to be one that is capable of energy/force prediction. Got {task_module.__type__}."
        if isinstance(task_module, MultiTaskLitModule):
            assert any(
                [
                    isinstance(
                        subtask,
                        (
                            ForceRegressionTask,
                            ScalarRegressionTask,
                            GradFreeForceRegressionTask,
                        ),
                    )
                    for subtask in task_module.task_list
                ]
            ), "Expected at least one subtask to be energy/force predictor."
        self.task_module = task_module
        self.transforms = transforms
        self.conversion_factor = conversion_factor

    @property
    def conversion_factor(self) -> dict[str, float]:
        return self._conversion_factor

    @conversion_factor.setter
    def conversion_factor(self, factor: float | dict[str, float]) -> None:
        if isinstance(factor, float):
            factor = {"energy": factor}
        for key in factor.keys():
            if key not in self.implemented_properties:
                raise KeyError(
                    f"Conversion factor {key} is not in `implemented_properties`."
                )
        self._conversion_factor = factor

    @property
    def dtype(self) -> torch.dtype | str:
        dtype = self.task_module.dtype
        return dtype

    def _format_atoms(self, atoms: Atoms) -> DataDict:
        data_dict = {}
        pos = torch.from_numpy(atoms.get_positions())
        atomic_numbers = torch.LongTensor(atoms.get_atomic_numbers())
        cell = torch.from_numpy(atoms.get_cell(complete=True).array)
        # add properties to data dict
        data_dict["pos"] = pos
        data_dict["atomic_numbers"] = atomic_numbers
        data_dict["cell"] = cell
        return data_dict

    def _format_pipeline(self, atoms: Atoms) -> DataDict:
        """
        Main function that takes an ``ase.Atoms`` object and gets it
        ready for matsciml model consumption.

        We call ``_format_atoms`` to get the data in a format that
        is similar to what comes out datasets implemented in matsciml,
        so that the remainder of the transform pipeline can be used
        to obtain nominally the same behavior as you would in the
        rest of the pipeline.
        """
        # initial formatting to get something akin to dataset outputs
        data_dict = self._format_atoms(atoms)
        # type cast into the type expected by the model
        data_dict = recursive_type_cast(
            data_dict, self.dtype, ignore_keys=["atomic_numbers"], convert_numpy=True
        )
        # now run through the same transform pipeline as for datasets
        if self.transforms:
            for transform in self.transforms:
                data_dict = transform(data_dict)
        return data_dict

    def calculate(
        self,
        atoms=None,
        properties: list[Literal["energy", "forces"]] = ["energy", "forces"],
        system_changes=...,
    ) -> None:
        # retrieve atoms even if not passed
        Calculator.calculate(self, atoms)
        # get into format ready for matsciml model
        data_dict = self._format_pipeline(atoms)
        # run the data structure through the model
        output = self.task_module(data_dict)
        # add outputs to self.results as expected by ase
        if "energy" in output:
            self.results["energy"] = output["energy"].detach().item()
        if "force" in output:
            self.results["forces"] = output["force"].detach().numpy()
        if "stress" in output:
            self.results["stress"] = output["stress"].detach().numpy()
        if "dipole" in output:
            self.results["dipole"] = output["dipole"].detach().numpy()
        if len(self.results) == 0:
            raise RuntimeError(
                f"No expected properties were written. Output dict: {output}"
            )
        # perform optional unit conversions
        for key, value in self.conversion_factor.items():
            if key in self.results:
                self.results[key] *= value

    @classmethod
    def from_pretrained_force_regression(
        cls, ckpt_path: str | Path, *args, **kwargs
    ) -> MatSciMLCalculator:
        ckpt_path = __checkpoint_conversion_exist_check(ckpt_path)
        task = ForceRegressionTask.load_from_checkpoint(ckpt_path)
        return cls(task, *args, **kwargs)

    @classmethod
    def from_pretrained_gradfree_task(
        cls, ckpt_path: str | Path, *args, **kwargs
    ) -> MatSciMLCalculator:
        ckpt_path = __checkpoint_conversion_exist_check(ckpt_path)
        task = GradFreeForceRegressionTask.load_from_checkpoint(ckpt_path)
        return cls(task, *args, **kwargs)

    @classmethod
    def from_pretrained_scalar_task(
        cls, ckpt_path: str | Path, *args, **kwargs
    ) -> MatSciMLCalculator:
        ckpt_path = __checkpoint_conversion_exist_check(ckpt_path)
        task = ScalarRegressionTask.load_from_checkpoint(ckpt_path)
        return cls(task, *args, **kwargs)
