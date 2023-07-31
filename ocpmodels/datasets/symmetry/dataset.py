from typing import Any, List, Tuple, Union, Dict, Optional, Callable
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import IterableDataset

from ocpmodels.datasets.base import BaseLMDBDataset
from ocpmodels.common.registry import registry
from ocpmodels.datasets.utils import point_cloud_featurization, concatenate_keys


class OTFPointGroupDataset(IterableDataset):
    """This implements a variant of the point group dataset that generates batches on the fly"""

    ...


@registry.register_dataset("SyntheticPointGroupDataset")
class SyntheticPointGroupDataset(BaseLMDBDataset):
    __devset__ = Path(__file__).parents[0].joinpath("devset")

    def __init__(
        self,
        lmdb_root_path: Union[str, Path],
        transforms: Optional[List[Callable]] = None,
        max_types: int = 200,
    ) -> None:
        super().__init__(lmdb_root_path, transforms)
        self.max_types = max_types

    def index_to_key(self, index: int) -> Tuple[int]:
        return (0, index)

    def data_from_key(
        self, lmdb_index: int, subindex: int
    ) -> Dict[str, Union[Dict[str, torch.Tensor], torch.Tensor]]:
        sample = super().data_from_key(lmdb_index, subindex)
        # coordinates remains the original particle positions
        coords = sample["coordinates"]
        pc_pos = coords[None, :] - coords[:, None]
        # remap to the same keys as other datasets
        sample["pos"] = pc_pos
        # have filler keys to pretend like other data
        sample["pc_features"] = point_cloud_featurization(
            sample["source_types"], sample["dest_types"], self.max_types
        )
        sample["symmetry"] = {"number": sample["label"].item()}
        sample["num_centers"] = len(sample["source_types"])
        sample["num_neighbors"] = len(sample["dest_types"])
        # clean up keys
        for key in ["label"]:
            del sample[key]
        sample["targets"] = []
        sample["target_keys"] = {"regression": [], "classification": []}
        return sample

    @staticmethod
    def collate_fn(
        batch: List[Dict[str, Union[Dict[str, torch.Tensor], torch.Tensor]]]
    ):
        pad_keys = ["pos", "atomic_numbers"]
        batched_data = concatenate_keys(batch, pad_keys)
        return batched_data
