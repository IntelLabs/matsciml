from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import IterableDataset

from matsciml.common.registry import registry
from matsciml.datasets.base import BaseLMDBDataset
from matsciml.datasets.utils import concatenate_keys, point_cloud_featurization


class OTFPointGroupDataset(IterableDataset):
    """This implements a variant of the point group dataset that generates batches on the fly"""

    ...


@registry.register_dataset("SyntheticPointGroupDataset")
class SyntheticPointGroupDataset(BaseLMDBDataset):
    __devset__ = Path(__file__).parents[0].joinpath("devset")

    def __init__(
        self,
        lmdb_root_path: str | Path,
        transforms: list[Callable] | None = None,
        max_types: int = 200,
    ) -> None:
        super().__init__(lmdb_root_path, transforms)
        self.max_types = max_types

    def data_from_key(
        self,
        lmdb_index: int,
        subindex: int,
    ) -> dict[str, dict[str, torch.Tensor] | torch.Tensor]:
        sample = super().data_from_key(lmdb_index, subindex)
        # remap to the same keys as other datasets
        sample["pos"] = sample["coordinates"]
        del sample["coordinates"]
        # here dest_types and source_types are seemingly swapped: the
        # point group seemed to match best w.r.t coordinates and atom types
        # this way, while source_types look random
        sample["pc_features"] = point_cloud_featurization(
            sample["dest_types"],
            sample["source_types"],
            self.max_types,
        )
        sample["symmetry"] = {"number": sample["label"].item()}
        # this is consistently sweapped as with the featurization
        sample["num_centers"] = len(sample["dest_types"])
        sample["num_neighbors"] = len(sample["source_types"])
        sample["src_mask"] = sample["dest_types"] != 0.0
        sample["dst_mask"] = sample["source_types"] != 0.0
        # get number of particles in the original system
        sample["sizes"] = len(sample["pos"])
        # these are meant to correspond to indices
        sample["src_nodes"] = torch.arange(sample["num_centers"])
        sample["dst_nodes"] = torch.arange(sample["num_centers"])
        # clean up keys
        for key in ["label"]:
            del sample[key]
        sample["targets"] = []
        sample["target_keys"] = {"regression": [], "classification": []}
        return sample

    @staticmethod
    def collate_fn(
        batch: list[dict[str, dict[str, torch.Tensor] | torch.Tensor]],
    ):
        pad_keys = ["pc_features"]
        batched_data = concatenate_keys(
            batch,
            pad_keys,
            unpacked_keys=["sizes", "num_centers", "num_neighbors"],
        )
        return batched_data
