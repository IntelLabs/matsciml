from typing import Any, List, Tuple, Union, Dict, Optional, Callable
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import IterableDataset

from ocpmodels.datasets.base import BaseLMDBDataset
from ocpmodels.common.registry import registry
from ocpmodels.datasets.utils import point_cloud_featurization


def concatenate_keys(
    batch: List[Dict[str, Any]], pad_keys: List[str] = []
) -> Dict[str, Union[Dict[str, torch.Tensor], torch.Tensor]]:
    """
    Function for concatenating data along keys within a dictionary.

    Acts as a generic concatenation function, which can also be recursively
    applied to subdictionaries. The result is a dictionary with the same
    structure as each sample within a batch, with the exception of
    `target_keys` and `targets`, which are left blank for this dataset.

    Parameters
    ----------
    batch : List[Dict[str, Any]]
        List of samples to concatenate
    pad_keys : List[str]
        List of keys that are singled out to apply `pad_sequence` to.
        This is used for atom-centered point clouds, where the number
        of centers may not be the same between samples.

    Returns
    -------
    Dict[str, Union[Dict[str, torch.Tensor], torch.Tensor]]
        Concatenated data, following the same structure as each sample
        within `batch`.
    """
    sample = batch[0]
    batched_data = {}
    for key, value in sample.items():
        if key not in ["targets", "target_keys"]:
            if isinstance(value, dict):
                # apply function recursively on dictionaries
                result = concatenate_keys([s[key] for s in batch])
            else:
                elements = [s[key] for s in batch]
                if isinstance(value, torch.Tensor):
                    # for tensors that need to be padded
                    if key in pad_keys:
                        result = pad_sequence(elements, batch_first=True)
                    else:
                        result = torch.vstack(elements)
                # for scalar values (typically labels) pack them
                elif isinstance(value, (float, int)):
                    result = torch.tensor(elements)
                # for everything else, just return a list
                else:
                    result = elements
            batched_data[key] = result
    for key in ["targets", "target_keys"]:
        if key in sample:
            batched_data[key] = sample[key]
    return batched_data


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
