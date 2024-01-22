from __future__ import annotations

import pickle
from collections.abc import Iterable
from copy import deepcopy
from functools import cache, cached_property
from importlib.util import find_spec
from math import pi
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from emmet.core.symmetry import SymmetryData
from matgl.ext.pymatgen import Structure2Graph
from matgl.graph.data import M3GNetDataset
from pymatgen.analysis import local_env
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.core import Lattice, Structure
from tqdm import tqdm

from matsciml.common.registry import registry
from matsciml.common.types import BatchDict, DataDict
from matsciml.datasets.base import PointCloudDataset
from matsciml.datasets.utils import (
    concatenate_keys,
    element_types,
    point_cloud_featurization,
)

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


@registry.register_dataset("MaterialsProjectDataset")
class MaterialsProjectDataset(PointCloudDataset):
    __devset__ = Path(__file__).parents[0].joinpath("devset")

    def index_to_key(self, index: int) -> tuple[int]:
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
        structure: None | Structure = data.get("structure", None)
        if structure is None:
            raise ValueError(
                "Structure not found in data - workflow needs a structure to use!",
            )
        coords = torch.from_numpy(structure.cart_coords).float()
        system_size = len(coords)
        return_dict["pos"] = coords
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

    @property
    def target_keys(self) -> dict[str, list[str]]:
        # This returns the standardized dictionary of task_type/key mapping.
        keys = getattr(self, "_target_keys", None)
        if not keys:
            # grab a sample from the data to set the keys
            _ = self.__getitem__(0)
        return self._target_keys

    @property
    def target_key_list(self) -> list[str] | None:
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
    def target_keys(self, values: dict[str, list[str]]) -> None:
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
        self,
        lmdb_index: int,
        subindex: int,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
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
        data: dict[str, Any] = super().data_from_key(lmdb_index, subindex)
        return_dict = {}
        # parse out relevant structure/lattice data
        self._parse_structure(data, return_dict)
        self._parse_symmetry(data, return_dict)
        # assume every other key are targets
        not_targets = set(
            ["structure", "symmetry", "fields_not_requested", "formula_pretty"]
            + data["fields_not_requested"],
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
        value: float | Iterable[float],
    ) -> torch.Tensor | float:
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
    def dataset_target_norm(self) -> tuple[torch.Tensor, torch.Tensor]:
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
    def collate_fn(batch: list[DataDict]) -> BatchDict:
        # since this class returns point clouds by default, we have to pad
        # the atom-centered point cloud data
        return concatenate_keys(
            batch,
            pad_keys=["pc_features"],
            unpacked_keys=["sizes", "src_nodes", "dst_nodes"],
        )


if _has_pyg:
    from torch_geometric.data import Batch, Data

    CrystalNN = local_env.CrystalNN(
        distance_cutoffs=None,
        x_diff_weight=-1,
        porous_adjustment=False,
    )

    @registry.register_dataset("PyGMaterialsProjectDataset")
    class PyGMaterialsProjectDataset(MaterialsProjectDataset):
        """
        Subclass of `MaterialsProjectDataset` that will emit PyG graphs.

        This class should be used until a future refactor to unify data
        structures, and a transform interface is created for PyG graph creation.
        )
        """

        def __init__(
            self,
            lmdb_root_path: str | Path,
            cutoff_dist: float = 5.0,
            transforms: list[Callable] | None = None,
        ) -> None:
            """
            Instantiate a `PyGMaterialsProjectDataset` object.

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
            self,
            lmdb_index: int,
            subindex: int,
        ) -> dict[str, torch.Tensor | Data | dict[str, torch.Tensor]]:
            """
            Maps a pair of indices to a specific data sample from LMDB.

            This method in particular wraps the parent's method, which emits
            a point cloud among other data from Materials Project. The additional
            steps here are to: 1) compute an adjacency list, 2) pack data into a
            PyG Data structure, 3) clean up redundant data.

            Parameters
            ----------
            lmdb_index : int
                Index corresponding to which LMDB environment to parse from.
            subindex : int
                Index within an LMDB file that maps to a sample.

            Returns
            -------
            Dict[str, Union[torch.Tensor, pyg.Data, Dict[str, torch.Tensor]]]
                Single data sample from Materials Project with a "graph" key
            """
            data = super().data_from_key(lmdb_index, subindex)
            dist_mat: np.ndarray = data.get("distance_matrix").numpy()
            lower_tri = np.tril(dist_mat)
            # mask out self loops and atoms that are too far away
            mask = (0.0 < lower_tri) * (lower_tri < self.cutoff_dist)
            adj_list = np.argwhere(mask).tolist()  # DGLGraph only takes lists
            # number of nodes has to be passed explicitly since cutoff
            # radius may result in shorter adj_list
            graph = Data(
                edge_index=torch.LongTensor(adj_list),
                num_nodes=len(data["atomic_numbers"]),
            )
            graph["pos"] = data.get("coords", data["pos"])
            graph["atomic_numbers"] = data["atomic_numbers"]
            data["graph"] = graph
            # delete the keys to reduce data redundancy
            for key in [
                "pos",
                "coords",
                "atomic_numbers",
                "distance_matrix",
                "pc_features",
            ]:
                try:
                    del data[key]
                except KeyError:
                    pass
            return data

        @staticmethod
        def collate_fn(
            batch: list[dict[str, torch.Tensor | dict[str, torch.Tensor]]],
        ) -> dict[str, torch.Tensor | Data | dict[str, torch.Tensor]]:
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
            batched_data = Batch.from_data_list(batch)
            return batched_data

    @registry.register_dataset("PyGCdvaeDataset")
    class PyGCdvaeDataset(PyGMaterialsProjectDataset):
        def __init__(
            self,
            lmdb_root_path: str | Path,
            cutoff_dist: float = 5.0,
            transforms: list[Callable] | None = None,
            max_atoms: int = 25,
        ) -> None:
            super().__init__(lmdb_root_path, cutoff_dist, transforms)

            self.max_atoms = max_atoms
            self.lattice_scaler = None
            self.scaler = None

        def data_from_key(
            self,
            lmdb_index: int,
            subindex: int,
        ) -> dict[str, torch.Tensor | Data | dict[str, torch.Tensor]]:
            data = super(PyGMaterialsProjectDataset, self).data_from_key(
                lmdb_index,
                subindex,
            )
            num_nodes = len(data["atomic_numbers"])
            if num_nodes > 25:
                return {}
            edge_index = data["edge_index"]
            lattice_params = data["lattice_features"]["lattice_params"]
            # Changed the following lines so tests would run.
            y = data["targets"].get("formation_energy_per_atom", None)
            # scale target property
            if y is not None:
                if self.scaler is not None:
                    prop = self.scaler.transform(y)
                else:
                    prop = torch.Tensor([y])
            else:
                prop = torch.Tensor([torch.nan])
            # (frac_coords, atom_types, lengths, angles, edge_indices,
            # to_jimages, num_atoms) = data_dict['graph_arrays']

            # atom_coords are fractional coordinates
            # edge_index is incremented during batching
            data = Data(
                frac_coords=torch.Tensor(data["frac_coords"]),
                atom_types=torch.LongTensor(data["atomic_numbers"]),
                lengths=torch.Tensor(lattice_params[:3]).view(1, -1),
                angles=torch.Tensor(lattice_params[3:]).view(1, -1),
                edge_index=edge_index,  # shape (2, num_edges)
                to_jimages=data["to_jimages"],
                num_atoms=len(data["atomic_numbers"]),
                num_bonds=edge_index.shape[1],
                num_nodes=len(data["atomic_numbers"]),
                y=prop.view(1, -1),
            )
            return data

        def _parse_structure(
            self,
            data: dict[str, Any],
            return_dict: dict[str, Any],
        ) -> None:
            """
            The same as OG with the addition of jimages field
            """
            structure: None | Structure = data.get("structure", None)
            if structure is None:
                raise ValueError(
                    "Structure not found in data - workflow needs a structure to use!",
                )
            coords = torch.from_numpy(structure.cart_coords).float()
            return_dict["pos"] = coords[None, :] - coords[:, None]
            return_dict["coords"] = coords
            return_dict["frac_coords"] = structure.frac_coords
            atom_numbers = torch.LongTensor(structure.atomic_numbers)
            return_dict["atomic_numbers"] = torch.LongTensor(structure.atomic_numbers)
            # return_dict["pc_features"] = pc_features
            return_dict["num_particles"] = len(atom_numbers)
            return_dict["distance_matrix"] = torch.from_numpy(
                structure.distance_matrix,
            ).float()

            crystal_graph = StructureGraph.with_local_env_strategy(structure, CrystalNN)
            edge_indices, to_jimages = [], []
            for i, j, to_jimage in crystal_graph.graph.edges(data="to_jimage"):
                edge_indices.append([j, i])
                to_jimages.append(to_jimage)
                edge_indices.append([i, j])
                to_jimages.append(tuple(-tj for tj in to_jimage))
            return_dict["to_jimages"] = torch.LongTensor(to_jimages)
            return_dict["edge_index"] = torch.LongTensor(edge_indices).T

            # grab lattice properties
            # SUPER SLOW
            abc = structure.lattice.abc
            angles = structure.lattice.angles
            lattice_params = torch.FloatTensor(
                abc + tuple(angles),
            )
            lattice_features = {
                "lattice_params": lattice_params,
            }
            return_dict["lattice_features"] = lattice_features

        @cache
        def _load_keys(self) -> list[tuple[int, int]]:
            """
            Load in all of the indices from each LMDB file. This creates an
            easy lookup of which data point is mapped to which total dataset
            index, as the former is returned as a simple 2-tuple of lmdb
            file index and the subindex (i.e. the actual data to read in).

            Returns
            -------
            List[Tuple[int, int]]
                2-tuple of LMDB file index and data index within the file
            """
            indices = []
            for lmdb_index, env in enumerate(self._envs):
                with env.begin() as txn:
                    # this gets all the keys within the LMDB file, including metadata
                    lmdb_keys = [
                        value.decode("utf-8")
                        for value in txn.cursor().iternext(values=False)
                    ]
                    # filter out non-numeric keys
                    subindices = filter(lambda x: x.isnumeric(), lmdb_keys)

                    for idx in tqdm(subindices):
                        item = pickle.loads(txn.get(f"{idx}".encode("ascii")))
                        num_atoms = len(item["structure"].atomic_numbers)
                        if num_atoms <= self.max_atoms:
                            indices.append((lmdb_index, int(idx)))
            return indices

    @registry.register_dataset("CdvaeLMDBDataset")
    class CdvaeLMDBDataset(PyGMaterialsProjectDataset):
        def __init__(
            self,
            lmdb_root_path: str | Path,
            cutoff_dist: float = 5.0,
            transforms: list[Callable] | None = None,
            max_atoms: int = 25,
        ) -> None:
            super().__init__(lmdb_root_path, cutoff_dist, transforms)

            self.max_atoms = max_atoms
            self.lattice_scaler = None
            self.scaler = None

        def data_from_key(
            self,
            lmdb_index: int,
            subindex: int,
        ) -> dict[str, torch.Tensor | Data | dict[str, torch.Tensor]]:
            # The LMDB dataset already has prepared PyG graphs
            data = super(MaterialsProjectDataset, self).data_from_key(
                lmdb_index,
                subindex,
            )
            return data

        def index_to_key(self, index: int) -> tuple[int]:
            """Look up the index number in the list of LMDB keys"""
            return self.keys[index]

        @cache
        def _load_keys(self) -> list[tuple[int, int]]:
            indices = []
            for lmdb_index, env in enumerate(self._envs):
                with env.begin() as txn:
                    # this gets all the keys within the LMDB file, including metadata
                    lmdb_keys = [
                        value.decode("utf-8")
                        for value in txn.cursor().iternext(values=False)
                    ]
                    # filter out non-numeric keys
                    subindices = filter(lambda x: x.isnumeric(), lmdb_keys)
                    indices.extend(
                        [(lmdb_index, int(subindex)) for subindex in subindices],
                    )
            return indices


@registry.register_dataset("M3GMaterialsProjectDataset")
class M3GMaterialsProjectDataset(MaterialsProjectDataset):
    def __init__(
        self,
        lmdb_root_path: str | Path,
        threebody_cutoff: float = 4.0,
        cutoff_dist: float = 20.0,
        graph_labels: list[int | float] | None = None,
        transforms: list[Callable[..., Any]] | None = None,
    ):
        super().__init__(lmdb_root_path, transforms)
        self.threebody_cutoff = threebody_cutoff
        self.graph_labels = graph_labels
        self.cutoff_dist = cutoff_dist

    def _parse_structure(
        self,
        data: dict[str, Any],
        return_dict: dict[str, Any],
    ) -> None:
        super()._parse_structure(data, return_dict)
        structure: None | Structure = data.get("structure", None)
        self.structures = [structure]
        self.converter = Structure2Graph(
            element_types=element_types(),
            cutoff=self.cutoff_dist,
        )
        graphs, lg, sa = M3GNetDataset.process(self)
        return_dict["graph"] = graphs[0]
