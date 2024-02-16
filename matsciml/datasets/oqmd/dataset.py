from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch

from matsciml.common.registry import registry
from matsciml.common.types import BatchDict, DataDict
from matsciml.datasets.base import PointCloudDataset
from matsciml.datasets.utils import (
    concatenate_keys,
    point_cloud_featurization,
)


@registry.register_dataset("OQMDDataset")
class OQMDDataset(PointCloudDataset):
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

    @property
    def target_keys(self) -> dict[str, list[str]]:
        """Specifies tasks and their target keys. If more labels are desired this is
        they should be added by hand.

        Returns:
            Dict[str, List[str]]: target keys
        """
        return {
            "regression": ["energy", "band_gap", "stability"],
        }

    def target_key_list(self):
        keys = []
        for k, v in self.target_keys.items():
            keys.extend(v)
        return keys

    def data_from_key(
        self,
        lmdb_index: int,
        subindex: int,
    ) -> dict[str, dict[str, torch.Tensor] | torch.Tensor]:
        """Available keys and their descriptions may be found here: https://static.oqmd.org/static/docs/restful.html#

        Parameters
        ----------
        lmdb_index : int
            lmdb file to select
        subindex : int
            index within lmdb file to select

        Returns
        -------
        Dict[str, Union[Dict[str, torch.Tensor], torch.Tensor]]
            output data that is used during training
        """
        data = super().data_from_key(lmdb_index, subindex)
        return_dict = {}
        # coordinates remains the original particle positions
        coords = torch.tensor(data["cart_coords"])
        return_dict["pos"] = coords
        unit_cell = data["unit_cell"]
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
        return_dict["unit_cell"] = unit_cell
        return_dict.update(**node_choices)

        # delta_e is formation energy
        return_dict["energy"] = data["delta_e"]
        return_dict["stability"] = data["stability"]
        return_dict["band_gap"] = data["band_gap"]

        target_keys = self.target_key_list()
        targets = {
            key: self._standardize_values(return_dict[key]) for key in target_keys
        }
        return_dict = {
            key: self._standardize_values(return_dict[key]) for key in return_dict
        }
        return_dict["targets"] = targets
        # only have spacegroup name
        return_dict["symmetry"] = {
            "name": data["spacegroup"],
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
