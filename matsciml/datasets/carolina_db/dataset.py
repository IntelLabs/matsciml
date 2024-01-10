from __future__ import annotations

from collections.abc import Iterable
from math import pi
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from matgl.ext.pymatgen import Structure2Graph
from matgl.graph.data import M3GNetDataset
from pymatgen.core import Lattice, Structure

from matsciml.common.registry import registry
from matsciml.common.types import BatchDict, DataDict
from matsciml.datasets.base import PointCloudDataset
from matsciml.datasets.utils import (
    atomic_number_map,
    concatenate_keys,
    element_types,
    pad_point_cloud,
    point_cloud_featurization,
)


@registry.register_dataset("CMDataset")
class CMDataset(PointCloudDataset):
    __devset__ = Path(__file__).parents[0].joinpath("devset")

    def index_to_key(self, index: int) -> tuple[int]:
        return (0, index)

    @staticmethod
    def collate_fn(batch: list[DataDict]) -> BatchDict:
        return concatenate_keys(
            batch,
            pad_keys=["pc_features"],
            unpacked_keys=["sizes", "src_nodes", "dst_nodes"],
        )

    def raw_sample(self, idx):
        return super().data_from_key(0, idx)

    def data_from_key(
        self,
        lmdb_index: int,
        subindex: int,
    ) -> dict[str, dict[str, torch.Tensor] | torch.Tensor]:
        """Available keys from Carolina Materials Database:
            _symmetry_space_group_name_H-M
            _cell_length_a
            _cell_length_b
            _cell_length_c
            _cell_angle_alpha
            _cell_angle_beta
            _cell_angle_gamma
            _symmetry_Int_Tables_number
            _chemical_formula_structural
            _chemical_formula_sum
            _cell_volume
            _cell_formula_units_Z
            symmetry_dict
            atomic_numbers
            cart_coords
            energy
            formula_pretty
            origin_file

        Args:
            lmdb_index (int): lmdb file to select from
            subindex (int): index within lmdb file to select

        Returns:
            Dict[str, Union[Dict[str, torch.Tensor], torch.Tensor]]: output data that
            is used during training
        """
        data = super().data_from_key(lmdb_index, subindex)
        return_dict = {}
        # coordinates remains the original particle positions
        coords = torch.tensor(data["cart_coords"])
        return_dict["pos"] = coords
        system_size = coords.size(0)
        node_choices = self.choose_dst_nodes(system_size, self.full_pairwise)
        src_nodes, dst_nodes = node_choices["src_nodes"], node_choices["dst_nodes"]
        atom_numbers = torch.LongTensor(data["atomic_numbers"])
        # uses one-hot encoding featurization
        pc_features = point_cloud_featurization(
            atom_numbers[src_nodes],
            atom_numbers[dst_nodes],
            100,
        )
        return_dict["atomic_numbers"] = atom_numbers
        return_dict["pc_features"] = pc_features
        return_dict["sizes"] = system_size
        return_dict.update(**node_choices)

        lattice_abc = (
            float(data["_cell_length_a"]),
            float(data["_cell_length_b"]),
            float(data["_cell_length_c"]),
        )
        lattice_angles = (
            float(data["_cell_angle_alpha"]),
            float(data["_cell_angle_beta"]),
            float(data["_cell_angle_gamma"]),
        )

        lattice_params = torch.FloatTensor(
            lattice_abc + tuple(a * (pi / 180.0) for a in lattice_angles),
        )
        return_dict["lattice_params"] = lattice_params

        return_dict["energy"] = data["energy"]
        target_keys = ["energy"]
        targets = {key: self._standardize_values(data[key]) for key in target_keys}
        return_dict = {
            key: self._standardize_values(return_dict[key]) for key in return_dict
        }
        return_dict["targets"] = targets
        # there is more symmetry data, not sure what to keep or discard.
        return_dict["symmetry"] = {
            "number": int(data["_symmetry_Int_Tables_number"]),
            "name": data["_symmetry_space_group_name_H-M"],
        }
        target_types = {"regression": [], "classification": []}
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

        return return_dict

    @property
    def target_keys(self) -> dict[str, list[str]]:
        return {"regression": ["energy"]}

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


@registry.register_dataset("M3GCMDataset")
class M3GCMDataset(CMDataset):
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

    def data_from_key(self, lmdb_index: int, subindex: int) -> Any:
        return_dict = super().data_from_key(lmdb_index, subindex)
        a, b, c, alpha, beta, gamma = return_dict["lattice_params"]
        num_to_element = dict(
            zip(atomic_number_map().values(), atomic_number_map().keys()),
        )
        elements = [
            num_to_element[int(idx.item())] for idx in return_dict["atomic_numbers"]
        ]
        lattice = Lattice.from_parameters(
            a,
            b,
            c,
            alpha * 180 / pi,
            beta * 180 / pi,
            gamma * 180 / pi,
        )
        structure = Structure(lattice, elements, return_dict["pos"])
        self.structures = [structure]
        self.converter = Structure2Graph(
            element_types=element_types(),
            cutoff=self.cutoff_dist,
        )
        graphs, lg, sa = M3GNetDataset.process(self)
        return_dict["graph"] = graphs[0]
        return return_dict
