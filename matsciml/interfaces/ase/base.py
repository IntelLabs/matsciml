from __future__ import annotations

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


class MatSciMLCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        task_module: ScalarRegressionTask
        | GradFreeForceRegressionTask
        | ForceRegressionTask,
        transforms: list[AbstractDataTransform | Callable] | None = None,
        restart=None,
        label=None,
        atoms: Atoms | None = None,
        directory=".",
        **kwargs,
    ):
        super().__init__(
            restart, label=label, atoms=atoms, directory=directory, **kwargs
        )
        self.task_module = task_module
        self.transforms = transforms

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
            self.results["energy"] = output["energy"].item()
        if "force" in output:
            self.results["forces"] = output["force"].numpy()
