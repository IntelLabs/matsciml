from typing import Tuple, Dict, List, Union, Any, Optional, Callable
from pathlib import Path

import torch
import numpy as np
from ocpmodels.common.types import BatchDict, DataDict

from ocpmodels.datasets.base import PointCloudDataset
from ocpmodels.datasets.utils import concatenate_keys, point_cloud_featurization
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


@registry.register_dataset("LiPSDataset")
class LiPSDataset(PointCloudDataset):
    __devset__ = Path(__file__).parents[0].joinpath("devset")

    def index_to_key(self, index: int) -> Tuple[int]:
        return (0, index)

    @staticmethod
    def collate_fn(batch: List[DataDict]) -> BatchDict:
        return concatenate_keys(batch, pad_keys=["force", "pc_features", "pos"])

    def data_from_key(
        self, lmdb_index: int, subindex: int
    ) -> Dict[str, Union[float, torch.Tensor, Dict[str, torch.Tensor]]]:
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
        data["pos"] = coords[None, :] - coords[:, None]
        data["coords"] = coords
        atom_numbers = torch.LongTensor(data["atomic_numbers"])
        # uses one-hot encoding featurization
        pc_features = point_cloud_featurization(atom_numbers, atom_numbers, 200)
        # keep atomic numbers for graph featurization
        data["atomic_numbers"] = atom_numbers
        data["pc_features"] = pc_features
        data["num_particles"] = len(atom_numbers)

        data["targets"] = {}
        data["target_types"] = {"regression": [], "classification": []}
        for key in ["energy", "force"]:
            data["targets"][key] = data.get(key)
            data["target_types"]["regression"].append(key)
        return data

    @property
    def target_keys(self) -> Dict[str, List[str]]:
        return {"regression": ["energy", "force"]}
