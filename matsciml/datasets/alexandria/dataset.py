from __future__ import annotations

from collections.abc import Iterable
from math import pi
from pathlib import Path
from typing import Any
import numpy as np
import torch
import warnings
from emmet.core.symmetry import SymmetryData
from pymatgen.core.structure import Structure
from typing import Callable
from matsciml.common.registry import registry
from matsciml.common.types import BatchDict, DataDict
from matsciml.datasets.base import PointCloudDataset
from matsciml.datasets.utils import (
    concatenate_keys,
    point_cloud_featurization,
)


@registry.register_dataset("AlexandriaDataset")
class AlexandriaDataset(PointCloudDataset):
    __devset__ = Path(__file__).parents[0].joinpath("devset")

    def __init__(
        self,
        lmdb_root_path: str | Path,
        transforms: list[Callable[..., Any]] | None = None,
        full_pairwise: bool = True,
    ) -> None:
        super().__init__(lmdb_root_path, transforms, full_pairwise)
        if self.transforms:
            for transform in self.transforms:
                if (
                    (
                        hasattr(transform, "cutoff_radius")
                        and transform.cutoff_radius > 15.0
                    )
                    or (
                        hasattr(transform, "cutoff_dist")
                        and transform.cutoff_dist > 15.0
                    )
                    or (
                        hasattr(transform, "adaptive_cutoff")
                        and transform.adaptive_cutoff > 15.0
                    )
                ):
                    warnings.warn(
                        f"Transform {transform} has a cutoff radius > 15.0 this will lead to wrong neighborlists for the two and one-dimensional datasets."
                    )

    @property
    def target_keys(self) -> dict[str, list[str]]:
        # This returns the standardized dictionary of task_type/key mapping.
        keys = getattr(self, "_target_keys", None)
        if not keys:
            # grab a sample from the data to set the keys
            sample = self.__getitem__(0)
            if self.is_preprocessed:
                self._target_keys = sample["target_types"]
        return self._target_keys

    @target_keys.setter
    def target_keys(self, target_keys: dict[str, dict[str, list[str]]]) -> None:
        self._target_keys = target_keys

    def _parse_structure(
        self,
        data: dict[str, Any],
        return_dict: dict[str, Any],
    ) -> None:
        """
        Parse the standardized Structure data and format into torch Tensors.

        This method also includes extracting lattice data as well.

        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary corresponding to the materials project data structure.
        return_dict : Dict[str, Any]
            Output dictionary that contains the training sample. Mutates
            in place.

        Raises
        ------
        ValueError:
            If `Structure` is not found in the data; currently the workflow
            intends for you to have the data structure in place.
        """
        structure: None | Structure = Structure.from_dict(data.get("structure", None))
        if structure is None:
            raise ValueError(
                "Structure not found in data - workflow needs a structure to use!",
            )
        coords = torch.from_numpy(structure.cart_coords).float()
        system_size = len(coords)
        return_dict["pos"] = coords
        cell = torch.from_numpy(structure.lattice.matrix.copy()).float()
        return_dict["cell"] = cell
        chosen_nodes = self.choose_dst_nodes(system_size, self.full_pairwise)
        src_nodes, dst_nodes = chosen_nodes["src_nodes"], chosen_nodes["dst_nodes"]
        atom_numbers = torch.LongTensor(structure.atomic_numbers)
        # uses one-hot encoding featurization
        pc_features = point_cloud_featurization(
            atom_numbers[src_nodes],
            atom_numbers[dst_nodes],
            100,
        )
        # keep atomic numbers for graph featurization
        return_dict["atomic_numbers"] = atom_numbers
        return_dict["pc_features"] = pc_features
        return_dict["sizes"] = system_size
        return_dict.update(**chosen_nodes)
        return_dict["distance_matrix"] = torch.from_numpy(
            structure.distance_matrix,
        ).float()
        # grab lattice properties
        space_group = structure.get_space_group_info()[-1]
        return_dict["natoms"] = len(atom_numbers)
        lattice_params = torch.FloatTensor(
            structure.lattice.abc
            + tuple(a * (pi / 180.0) for a in structure.lattice.angles),
        )
        lattice_features = {
            "space_group": space_group,
            "lattice_params": lattice_params,
        }
        return_dict["lattice_features"] = lattice_features

    def _parse_symmetry(
        self,
        data: dict[str, Any],
        return_dict: dict[str, Any],
    ) -> None:
        """
        Parse out symmetry information from the `SymmetryData` structure.

        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary corresponding to the materials project data structure.
        return_dict : Dict[str, Any]
            Output dictionary that contains the training sample. Mutates
            in place.
        """
        symmetry: SymmetryData | None = data.get("symmetry", None)
        if symmetry is None:
            return
        else:
            symmetry_data = {
                "number": symmetry.number,
                "symbol": symmetry.symbol,
                "group": symmetry.point_group,
            }
            return_dict["symmetry"] = symmetry_data

    def data_from_key(
        self,
        lmdb_index: int,
        subindex: int,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """
        Extract data out of the PyMatGen data structure and format into PyTorch
        happy structures.

        In line with the rest of the framework, this method returns a nested
        dictionary. Specific to this format, however, we separate features and
        targets: at the top of level we expect what is effectively a point cloud
        with `coords` and `atomic_numbers`, while the `lattice_features` and
        `targets` keys nest additional values.

        This method is likely what you would want to change if you want to modify
        featurization for this project.

        Parameters
        ----------
        lmdb_index : int
            Index corresponding to which LMDB environment to parse from.
        subindex : int
            Index within an LMDB file that maps to a sample.

        Returns
        -------
        Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]
            A single sample/material from Materials Project.
        """
        data: dict[str, Any] = super().data_from_key(lmdb_index, subindex)
        return_dict = {"entry_id": data["entry_id"]}
        # parse out relevant structure/lattice data
        self._parse_structure(data, return_dict)
        self._parse_symmetry(data, return_dict)
        return_dict["force"] = self._standardize_values(data["force"])
        return_dict["magmoms"] = self._standardize_values(data["magmoms"])
        regression_targets = {
            key: self._standardize_values(data["targets"]["regression"][key])
            for key in data["targets"].get("regression", {}).keys()
        }
        classification_targets = {
            key: self._standardize_values(data["targets"]["classification"][key])
            for key in data["targets"].get("classification", {}).keys()
        }
        regression_targets.update(classification_targets)
        targets = regression_targets
        return_dict["targets"] = targets
        target_types = {
            "classification": list(data["targets"].get("classification", {}).keys()),
            "regression": list(data["targets"]["regression"]),
        }
        return_dict["target_types"] = target_types
        self.target_keys = target_types
        return return_dict

    @staticmethod
    def _standardize_values(
        value: float | Iterable[float],
    ) -> torch.Tensor | float:
        """
        Standardizes targets to be ingested by a model.

        For scalar values, we simply return it unmodified, because they can be easily
        collated. For iterables such as tuples and NumPy arrays, we use the appropriate
        tensor creation method, and typecasted into FP32 or Long tensors.

        The type hint `float` is used to denote numeric types more broadly.

        Parameters
        ----------
        value : Union[float, Iterable[float]]
            A target value, which can be a scalar or array of values

        Returns
        -------
        Union[torch.Tensor, float]
            Mapped torch.Tensor format, or a scalar numeric value

        Raises
        ------
        ValueError
            If the target data value is missing.
        """
        if isinstance(value, Iterable) and not isinstance(value, str):
            # get type from first entry
            dtype = torch.long if isinstance(value[0], int) else torch.float
            if isinstance(value, np.ndarray):
                return torch.from_numpy(value).type(dtype)
            else:
                return torch.Tensor(value).type(dtype)
        # for missing data, raise an error
        elif value is None:
            raise ValueError("Missing data!")
        else:
            # for scalars, just return the value
            return value

    @staticmethod
    def collate_fn(batch: list[DataDict]) -> BatchDict:
        # since this class returns point clouds by default, we have to pad
        # the atom-centered point cloud data
        return concatenate_keys(
            batch,
            pad_keys=["pc_features"],
            unpacked_keys=["sizes", "src_nodes", "dst_nodes"],
        )
