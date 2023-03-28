from typing import Iterable, Tuple, Any, Dict, Union, Optional, List, Callable
from importlib.util import find_spec
from pathlib import Path

import torch
import numpy as np
from pymatgen.core import Structure

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
        data = getattr(data, key, None)
        if data is None:
            raise KeyError(f"{key} not found in {data}.")
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
        structure: Structure = data.get("structure")
        # retrieve properties
        return_dict["pos"] = torch.from_numpy(structure.cart_coords).float()
        return_dict["atomic_numbers"] = torch.LongTensor(structure.atomic_numbers)
        return_dict["distance_matrix"] = torch.from_numpy(
            structure.distance_matrix
        ).float()
        # grab lattice properties
        space_group = structure.get_space_group_info()[-1]
        lattice_params = torch.FloatTensor(
            structure.lattice.abc + structure.lattice.angles
        )
        lattice_features = {
            "space_group": space_group,
            "lattice_params": lattice_params,
        }
        return_dict["lattice_features"] = lattice_features
        # assume every other key are targets
        not_targets = set(
            ["structure", "fields_not_requested"] + data["fields_not_requested"]
        )
        target_keys = set(data.keys()).difference(not_targets)
        targets = {key: self._standardize_values(data[key]) for key in target_keys}
        return_dict["targets"] = targets
        # compress all the targets into a single tensor for convenience
        target_tensor = []
        for key in target_keys:
            item = data.get(key)
            if isinstance(item, Iterable):
                target_tensor.extend(item)
            else:
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
