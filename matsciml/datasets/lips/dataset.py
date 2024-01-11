from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from matsciml.common.registry import registry
from matsciml.common.types import BatchDict, DataDict
from matsciml.datasets.base import PointCloudDataset
from matsciml.datasets.utils import concatenate_keys, point_cloud_featurization


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


@registry.register_dataset("LiPSDataset")
class LiPSDataset(PointCloudDataset):
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

    def data_from_key(
        self,
        lmdb_index: int,
        subindex: int,
    ) -> dict[str, float | torch.Tensor | dict[str, torch.Tensor]]:
        """
        Retrieve a sample from the LMDB file.

        The only modification here from the base method is to tack on some
        metadata about expected targets: for this dataset, we have energy
        and force labels.

        Parameters
        ----------
        lmdb_index : int
            Index of the LMDB file
        subindex : int
            Index of the sample within the LMDB file

        Returns
        -------
        Dict[str, Union[float, torch.Tensor, Dict[str, torch.Tensor]]]
            A single sample from this dataset
        """
        data = super().data_from_key(lmdb_index, subindex)

        coords = data["pos"]
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
        # keep atomic numbers for graph featurization
        data["atomic_numbers"] = atom_numbers
        data["pc_features"] = pc_features
        data["sizes"] = system_size
        data.update(**node_choices)

        data["targets"] = {}
        data["target_types"] = {"regression": [], "classification": []}
        for key in ["energy", "force"]:
            data["targets"][key] = data.get(key)
            data["target_types"]["regression"].append(key)
        return data

    @property
    def target_keys(self) -> dict[str, list[str]]:
        return {"regression": ["energy", "force"]}
