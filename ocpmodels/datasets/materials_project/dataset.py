from functools import cached_property
from typing import Iterable, Tuple, Any, Dict, Union, Optional, List, Callable
from pathlib import Path
from math import pi
from copy import deepcopy
from functools import cache
from tqdm import tqdm

import torch
import numpy as np
import pickle
from pymatgen.core import Structure
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis import local_env
from emmet.core.symmetry import SymmetryData
from ocpmodels.common.types import BatchDict, DataDict

from ocpmodels.datasets.base import PointCloudDataset
from ocpmodels.datasets.utils import (
    concatenate_keys,
    point_cloud_featurization,
)
from ocpmodels.common.registry import registry


def item_from_structure(data: Any, *keys: str) -> Any:
    """
    Function to recurse through an object and retrieve a nested attribute.

    Parameters
    ----------
    data : Any
        Basically any Python object
    keys : str
        Any variable number of keys to recurse through.

    Returns
    -------
    Any
        Retrieved nested attribute/object

    Raises
    ------
    KeyError
        If a key is not present, raise KeyError.
    """
    for key in keys:
        assert key in data.keys(), f"{key} not found in {data}."
        if isinstance(data, dict):
            data = data.get(key)
        else:
            data = getattr(data, key)
    return data


@registry.register_dataset("MaterialsProjectDataset")
class MaterialsProjectDataset(PointCloudDataset):
    __devset__ = Path(__file__).parents[0].joinpath("devset")

    def index_to_key(self, index: int) -> Tuple[int]:
        """
        Method that maps a global index value to a pair of indices.

        This is kept for consistency between other datasets (namely
        OCP). For now we are not splitting the dataset to multiple
        LMDB files, so the first index that is returned is hardcoded.

        Parameters
        ----------
        index : int
            Global data sample index

        Returns
        -------
        Tuple[int]
            2-tuple of LMDB index and subindex
        """
        return (0, index)

    def _parse_structure(
        self, data: Dict[str, Any], return_dict: Dict[str, Any]
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
        structure: Union[None, Structure] = data.get("structure", None)
        if structure is None:
            raise ValueError(
                "Structure not found in data - workflow needs a structure to use!"
            )
        coords = torch.from_numpy(structure.cart_coords).float()
        system_size = len(coords)
        return_dict["pos"] = coords
        chosen_nodes = self.choose_dst_nodes(system_size, self.full_pairwise)
        src_nodes, dst_nodes = chosen_nodes["src_nodes"], chosen_nodes["dst_nodes"]
        atom_numbers = torch.LongTensor(structure.atomic_numbers)
        # uses one-hot encoding featurization
        pc_features = point_cloud_featurization(
            atom_numbers[src_nodes], atom_numbers[dst_nodes], 100
        )
        # keep atomic numbers for graph featurization
        return_dict["atomic_numbers"] = atom_numbers
        return_dict["pc_features"] = pc_features
        return_dict["sizes"] = system_size
        return_dict.update(**chosen_nodes)
        return_dict["distance_matrix"] = torch.from_numpy(
            structure.distance_matrix
        ).float()
        # grab lattice properties
        space_group = structure.get_space_group_info()[-1]
        # convert lattice angles into radians
        lattice_params = torch.FloatTensor(
            structure.lattice.abc
            + tuple(a * (pi / 180.0) for a in structure.lattice.angles)
        )
        lattice_features = {
            "space_group": space_group,
            "lattice_params": lattice_params,
        }
        return_dict["lattice_features"] = lattice_features

    def _parse_symmetry(
        self, data: Dict[str, Any], return_dict: Dict[str, Any]
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
        symmetry: Union[SymmetryData, None] = data.get("symmetry", None)
        if symmetry is None:
            return
        else:
            symmetry_data = {
                "number": symmetry.number,
                "symbol": symmetry.symbol,
                "group": symmetry.point_group,
            }
            return_dict["symmetry"] = symmetry_data

    @property
    def target_keys(self) -> Dict[str, List[str]]:
        # This returns the standardized dictionary of task_type/key mapping.
        keys = getattr(self, "_target_keys", None)
        if not keys:
            # grab a sample from the data to set the keys
            _ = self.__getitem__(0)
        return self._target_keys

    @property
    def target_key_list(self) -> Union[List[str], None]:
        # this returns a flat list of keys, primarily used in the `data_from_key`
        # call. This does not provide the task type/key mapping used to initialize
        # output heads
        keys = getattr(self, "_target_keys", None)
        if not keys:
            return keys
        else:
            _keys = []
            for key_list in keys.values():
                _keys.extend(key_list)
            return _keys

    @target_keys.setter
    def target_keys(self, values: Dict[str, List[str]]) -> None:
        remove_keys = []
        copy_dict = deepcopy(values)
        # loop over the keys and remove empty tasks
        for key, value in values.items():
            if len(value) == 0:
                remove_keys.append(key)
        for key in remove_keys:
            del copy_dict[key]
        self._target_keys = copy_dict

    def data_from_key(
        self, lmdb_index: int, subindex: int
    ) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Extract data out of the PyMatGen data structure and format into PyTorch happy structures.

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
        data: Dict[str, Any] = super().data_from_key(lmdb_index, subindex)
        return_dict = {}
        # parse out relevant structure/lattice data
        self._parse_structure(data, return_dict)
        self._parse_symmetry(data, return_dict)
        # assume every other key are targets
        not_targets = set(
            ["structure", "symmetry", "fields_not_requested", "formula_pretty"]
            + data["fields_not_requested"]
        )
        # target_keys = getattr(self, "_target_keys", None)
        target_keys = self.target_key_list
        # in the event we're getting data for the first time
        if not target_keys:
            target_keys = set(data.keys()).difference(not_targets)
            # cache the result
            target_keys = list(target_keys)
        targets = {key: self._standardize_values(data[key]) for key in target_keys}
        return_dict["targets"] = targets
        # compress all the targets into a single tensor for convenience
        target_types = {"classification": [], "regression": []}
        for key in target_keys:
            item = targets.get(key)
            if isinstance(item, Iterable):
                # check if the data is numeric first
                if isinstance(item[0], (float, int)):
                    target_types["regression"].append(key)
            else:
                if isinstance(item, (float, int)):
                    target_type = (
                        "classification" if isinstance(item, int) else "regression"
                    )
                    target_types[target_type].append(key)
        return_dict["target_types"] = target_types
        self.target_keys = target_types
        return return_dict

    @staticmethod
    def _standardize_values(
        value: Union[float, Iterable[float]]
    ) -> Union[torch.Tensor, float]:
        """
        Standardizes targets to be ingested by a model.

        For scalar values, we simply return it unmodified, because they can be easily collated.
        For iterables such as tuples and NumPy arrays, we use the appropriate tensor creation
        method, and typecasted into FP32 or Long tensors.

        The type hint `float` is used to denote numeric types more broadly.

        Parameters
        ----------
        value : Union[float, Iterable[float]]
            A target value, which can be a scalar or array of values

        Returns
        -------
        Union[torch.Tensor, float]
            Mapped torch.Tensor format, or a scalar numeric value
        """
        if isinstance(value, Iterable) and not isinstance(value, str):
            # get type from first entry
            dtype = torch.long if isinstance(value[0], int) else torch.float
            if isinstance(value, np.ndarray):
                return torch.from_numpy(value).type(dtype)
            else:
                return torch.Tensor(value).type(dtype)
        # for missing data, set to zero
        elif value is None:
            return 0.0
        else:
            # for scalars, just return the value
            return value

    @cached_property
    def dataset_target_norm(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the dataset average for targets.

        This property is cached, so once it has been called (per session)
        it should stash the result for performance.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            PyTorch tensor containing the mean and standard deviation
            values of targets, with `keepdim=True`
        """
        targets = [self.__getitem__(i)["target_tensor"] for i in range(len(self))]
        targets = torch.vstack(targets)
        return (targets.mean(0, keepdim=True), targets.std(0, keepdim=True))

    @staticmethod
    def collate_fn(batch: List[DataDict]) -> BatchDict:
        # since this class returns point clouds by default, we have to pad
        # the atom-centered point cloud data
        return concatenate_keys(
            batch,
            pad_keys=["pc_features"],
            unpacked_keys=["sizes", "src_nodes", "dst_nodes"],
        )
