from functools import cached_property
from typing import Iterable, Tuple, Any, Dict, Union, Optional, List, Callable
from importlib.util import find_spec
from pathlib import Path
from math import pi

import torch
import numpy as np
from pymatgen.core import Structure
from emmet.core.symmetry import SymmetryData

from ocpmodels.datasets.base import BaseOCPDataset


_has_dgl = find_spec("dgl") is not None
_has_pyg = find_spec("torch_geometric") is not None


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


class MaterialsProjectDataset(BaseOCPDataset):
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
        return_dict["pos"] = torch.from_numpy(structure.cart_coords).float()
        return_dict["atomic_numbers"] = torch.LongTensor(structure.atomic_numbers)
        return_dict["distance_matrix"] = torch.from_numpy(
            structure.distance_matrix
        ).float()
        # grab lattice properties
        space_group = structure.get_space_group_info()[-1]
        # convert lattice angles into radians
        lattice_params = torch.FloatTensor(
            structure.lattice.abc + tuple(a * (pi / 180.) for a in structure.lattice.angles)
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
    def target_keys(self) -> List[str]:
        return self._target_keys

    @target_keys.setter
    def target_keys(self, values: List[str]) -> None:
        self._target_keys = values

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
            ["structure", "symmetry", "fields_not_requested"]
            + data["fields_not_requested"]
        )
        target_keys = getattr(self, "target_keys", None)
        # in the event we're getting data for the first time
        if not target_keys:
            target_keys = set(data.keys()).difference(not_targets)
            # cache the result
            self.target_keys = list(target_keys)
        targets = {key: self._standardize_values(data[key]) for key in target_keys}
        return_dict["targets"] = targets
        # compress all the targets into a single tensor for convenience
        target_tensor = []
        for key in target_keys:
            item = data.get(key)
            if isinstance(item, Iterable):
                # check if the data is numeric first
                if isinstance(item[0], (float, int)):
                    target_tensor.extend(item)
            else:
                # big warning: if property is missing, we set the value to zero
                # TODO think about whether this is physical
                if item is None:
                    item = 0.0
                if isinstance(item, (float, int)):
                    target_tensor.append(item)
        target_tensor = torch.FloatTensor(target_tensor)
        return_dict["target_tensor"] = target_tensor
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
    def collate_fn(
        batch: List[Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]]
    ) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Relatively verbose function to batch up materials project data.

        At a high level, we basically go through and check what data types
        are needed for the full batch, based on the first sample, and use
        that to determine what to do: tensors are stacked, 1D lists are
        converted into their appropriate tensor types. For dictionaries,
        we do the same thing but at their level (i.e. we preserve the
        original structure of a sample). For strings, we are not currently
        doing anything with them and so they are preserved as lists of strings.

        We define a `pad_keys` list of keys that are explicitly meant to
        be padded, which correspond to point cloud entities. For memory
        considerations, the interatomic distance matrix is not batched
        and just left as a list of tensors.

        Parameters
        ----------
        batch : List[Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]]
            List of Materials Project samples

        Returns
        -------
        Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]
            Dictionary of batched data
        """
        joint_data = {}
        sample = batch[0]
        pad_keys = ["pos", "atomic_numbers"]
        # get the biggest point cloud size for padding
        if any([key in sample.keys() for key in pad_keys]):
            max_size = max([s["pos"].size(0) for s in batch])
            batch_size = len(batch)
        for key, value in sample.items():
            # for dictionaries, we need to go one level deeper
            if isinstance(value, dict):
                joint_data[key] = {}
                for subkey, subvalue in value.items():
                    data = [item_from_structure(s, key, subkey) for s in batch]
                    # for numeric types, cast to a float tensor
                    if isinstance(subvalue, (int, float)):
                        data = torch.as_tensor(data)
                    elif isinstance(subvalue, torch.Tensor):
                        data = torch.vstack(data)
                    # for string types, we just return a list
                    else:
                        pass
                    joint_data[key][subkey] = data
            elif key in pad_keys:
                assert isinstance(
                    value, torch.Tensor
                ), f"{key} in batch should be a tensor."
                # get the dimensionality
                tensor_dim = value.size(-1)
                if value.ndim == 2:
                    data = torch.zeros(
                        batch_size, max_size, tensor_dim, dtype=value.dtype
                    )
                else:
                    data = torch.zeros(batch_size, max_size, dtype=value.dtype)
                # pack the batched tensor with each sample now
                for index, s in enumerate(batch):
                    tensor: torch.Tensor = s[key]
                    if tensor.ndim == 2:
                        lengths = tensor.shape
                        data[index, : lengths[0], : lengths[1]] = tensor[:, :]
                    else:
                        length = len(tensor)
                        data[index, :length] = tensor[:]
                joint_data[key] = data
            else:
                # aggregate all the data
                data = [s.get(key) for s in batch]
                if isinstance(value, torch.Tensor) and key != "distance_matrix":
                    data = torch.vstack(data)
                elif isinstance(value, (int, float)):
                    data = torch.as_tensor(data)
                # return anything else as just a list
                joint_data[key] = data
        return joint_data


if _has_dgl:
    import dgl

    class DGLMaterialsProjectDataset(MaterialsProjectDataset):
        """
        Subclass of `MaterialsProjectDataset` that will emit DGL graphs.

        This class should be used until a future refactor to unify data
        structures, and a transform interface is created for DGL graph creation.
        )
        """

        def __init__(
            self,
            lmdb_root_path: Union[str, Path],
            cutoff_dist: float = 5.0,
            transforms: Optional[List[Callable]] = None,
        ) -> None:
            """
            Instantiate a `DGLMaterialsProjectDataset` object.

            In addition to specifying an optional list of transforms and an
            LMDB path, the `cutoff_dist` parameter is used to control edge
            creation: we take a point cloud structure and create edges for
            all atoms/sites that are within this cut off distance.

            Parameters
            ----------
            lmdb_root_path : Union[str, Path]
                Path to a folder containing LMDB files for Materials Project.
            cutoff_dist : float
                Distance to cut off edge creation; interatomic distances greater
                than this value will not have an edge.
            transforms : Optional[List[Callable]], by default None
                List of transforms to apply to the data.
            """
            super().__init__(lmdb_root_path, transforms)
            self.cutoff_dist = cutoff_dist

        @property
        def cutoff_dist(self) -> float:
            return self._cutoff_dist

        @cutoff_dist.setter
        def cutoff_dist(self, value: float) -> None:
            """
            Setter method for the cut off distance property.

            For now this doesn't do anything special, but can be modified
            to include checks for valid value, etc.

            Parameters
            ----------
            value : float
                Value to set the cut off distance
            """
            self._cutoff_dist = value

        def data_from_key(
            self, lmdb_index: int, subindex: int
        ) -> Dict[str, Union[torch.Tensor, dgl.DGLGraph, Dict[str, torch.Tensor]]]:
            """
            Maps a pair of indices to a specific data sample from LMDB.

            This method in particular wraps the parent's method, which emits
            a point cloud among other data from Materials Project. The additional
            steps here are to: 1) compute an adjacency list, 2) pack data into a
            DGLGraph structure, 3) clean up redundant data.

            Parameters
            ----------
            lmdb_index : int
                Index corresponding to which LMDB environment to parse from.
            subindex : int
                Index within an LMDB file that maps to a sample.

            Returns
            -------
            Dict[str, Union[torch.Tensor, dgl.DGLGraph, Dict[str, torch.Tensor]]]
                Single data sample from Materials Project with a "graph" key
            """
            data = super().data_from_key(lmdb_index, subindex)
            dist_mat: np.ndarray = data.get("distance_matrix").numpy()
            lower_tri = np.tril(dist_mat)
            # mask out self loops and atoms that are too far away
            mask = (0.0 < lower_tri) * (lower_tri < self.cutoff_dist)
            adj_list = np.argwhere(mask).tolist()  # DGLGraph only takes lists
            graph = dgl.graph(adj_list)
            graph.ndata["pos"] = data["pos"]
            graph.ndata["atomic_numbers"] = data["atomic_numbers"]
            data["graph"] = graph
            # delete the keys to reduce data redundancy
            for key in ["pos", "atomic_numbers", "distance_matrix"]:
                del data[key]
            return data

        @staticmethod
        def collate_fn(
            batch: List[Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]]
        ) -> Dict[str, Union[torch.Tensor, dgl.DGLGraph, Dict[str, torch.Tensor]]]:
            """
            Collate function for DGLGraph variant of the Materials Project.

            Basically uses the same workflow as that for `MaterialsProjectDataset`,
            but with the added step of calling `dgl.batch` on the graph data
            that is left unprocessed by the parent method.

            Parameters
            ----------
            batch : List[Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]]
                List of Materials Project samples

            Returns
            -------
            Dict[str, Union[torch.Tensor, dgl.DGLGraph, Dict[str, torch.Tensor]]]
                Batched data, including graph
            """
            batched_data = super(
                DGLMaterialsProjectDataset, DGLMaterialsProjectDataset
            ).collate_fn(batch)
            batched_data["graph"] = dgl.batch(batched_data["graph"])
            return batched_data
