from copy import deepcopy
from functools import cached_property
from math import pi
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch

from ocpmodels.common.registry import registry
from ocpmodels.common.types import BatchDict, DataDict
from ocpmodels.datasets.base import PointCloudDataset
from ocpmodels.datasets.utils import (
    concatenate_keys,
    pad_point_cloud,
    point_cloud_featurization,
)


@registry.register_dataset("NomadDataset")
class NomadDataset(PointCloudDataset):
    __devset__ = Path(__file__).parents[0].joinpath("devset")

    def index_to_key(self, index: int) -> Tuple[int]:
        return (0, index)

    @staticmethod
    def collate_fn(batch: List[DataDict]) -> BatchDict:
        return concatenate_keys(
            batch,
            pad_keys=["pc_features"],
            unpacked_keys=["sizes", "src_nodes", "dst_nodes"],
        )

    def raw_sample(self, idx):
        return super().data_from_key(0, idx)

    @property
    def target_keys(self) -> Dict[str, List[str]]:
        return {
            "regression": ["energy_total", "efermi"],
            "classification": ["spin_polarized"],
        }

    def target_key_list(self):
        keys = []
        for k, v in self.target_keys.items():
            keys.extend(v)
        return keys

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
    def atomic_number_map(self) -> Dict[str, int]:
        """List of element symbols and their atomic numbers.

        Returns:
            Dict[str, int]: _description_
        """
        # fmt: off
        an_map = {'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 
              'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 
              'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22, 
              'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 
              'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36, 
              'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42, 'Tc': 43, 
              'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50, 
              'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 
              'Ce': 58, 'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 
              'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71, 
              'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 
              'Au': 79, 'Hg': 80, 'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85, 
              'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90, 'Pa': 91, 'U': 92, 
              'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99, 
              'Fm': 100, 'Md': 101, 'No': 102, 'Lr': 103, 'Rf': 104, 'Db': 105, 
              'Sg': 106, 'Bh': 107, 'Hs': 108, 'Mt': 109, 'Ds': 110, 'Rg': 111, 
              'Cn': 112, 'Nh': 113, 'Fl': 114, 'Mc': 115, 'Lv': 116, 'Ts': 117, 
              'Og': 118}
        # fmt: on
        return an_map

    def _parse_data(self, data: Dict[str, Any], return_dict: Dict[str, Any]) -> None:
        cart_coords = torch.Tensor(
            data["properties"]["structures"]["structure_original"][
                "cartesian_site_positions"
            ]
        )
        system_size = len(cart_coords)
        return_dict["pos"] = cart_coords
        chosen_nodes = self.choose_dst_nodes(system_size, self.full_pairwise)
        src_nodes, dst_nodes = chosen_nodes["src_nodes"], chosen_nodes["dst_nodes"]

        atomic_numbers = torch.LongTensor(
            [
                self.atomic_number_map[symbol]
                for symbol in data["properties"]["structures"]["structure_original"][
                    "species_at_sites"
                ]
            ]
        )
        return_dict["atomic_numbers"] = atomic_numbers
        return_dict["cart_coords"] = cart_coords
        # uses one-hot encoding featurization
        pc_features = point_cloud_featurization(
            atomic_numbers[src_nodes], atomic_numbers[dst_nodes], 100
        )
        # keep atomic numbers for graph featurization
        return_dict["pc_features"] = pc_features
        return_dict["sizes"] = system_size
        return_dict.update(**chosen_nodes)

        # space_group = structure.get_space_group_info()[-1]
        # # convert lattice angles into radians
        lattice_a = data["properties"]["structures"]["structure_original"][
            "lattice_parameters"
        ]["a"]
        lattice_b = data["properties"]["structures"]["structure_original"][
            "lattice_parameters"
        ]["b"]
        lattice_c = data["properties"]["structures"]["structure_original"][
            "lattice_parameters"
        ]["c"]
        lattice_alpha = data["properties"]["structures"]["structure_original"][
            "lattice_parameters"
        ]["alpha"]
        lattice_beta = data["properties"]["structures"]["structure_original"][
            "lattice_parameters"
        ]["beta"]
        lattice_gamma = data["properties"]["structures"]["structure_original"][
            "lattice_parameters"
        ]["gamma"]
        lattice_abc = (lattice_a, lattice_b, lattice_c)
        lattice_angles = (lattice_alpha, lattice_beta, lattice_gamma)
        # Need to check if angles are in rad or deg
        lattice_params = torch.FloatTensor(
            lattice_abc + tuple(a * (pi / 180.0) for a in lattice_angles)
        )
        return_dict["lattice_params"] = lattice_params
        return_dict["efermi"] = data["properties"]["electronic"][
            "band_structure_electronic"
        ]["energy_fermi"]
        return_dict["energy_total"] = data["energies"]["total"]["value"]
        # data['properties']['electronic']['dos_electronic']['energy_fermi']
        return_dict["spin_polarized"] = data["properties"]["electronic"][
            "band_structure_electronic"
        ]["spin_polarized"]
        return_dict["symmetry"] = {}
        return_dict["symmetry"]["number"] = data["material"]["symmetry"][
            "space_group_number"
        ]
        return_dict["symmetry"]["symbol"] = data["material"]["symmetry"][
            "space_group_symbol"
        ]
        return_dict["symmetry"]["group"] = data["material"]["symmetry"]["point_group"]

        standard_keys = set(return_dict.keys()).difference(
            ["symmetry", "spin_polarized"]
        )
        standard_dict = {
            key: self._standardize_values(return_dict[key]) for key in standard_keys
        }
        return_dict.update(standard_dict)
        target_keys = self.target_key_list()
        targets = {
            key: self._standardize_values(return_dict[key]) for key in target_keys
        }
        return_dict["targets"] = targets

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

    def data_from_key(self, lmdb_index: int, subindex: int) -> Any:
        # for a full list of properties avaialbe: data['properties']['available_properties'
        data = super().data_from_key(lmdb_index, subindex)
        return_dict = {}
        self._parse_data(data, return_dict=return_dict)
        return return_dict
