from typing import Any, List, Tuple, Union, Dict

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import IterableDataset, DataLoader

from ocpmodels.datasets.base import BaseLMDBDataset


def concatenate_keys(
    batch: List[Dict[str, Any]], pad_keys: List[str] = []
) -> Dict[str, Union[Dict[str, torch.Tensor], torch.Tensor]]:
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


class PointGroupDataset(BaseLMDBDataset):
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
        sample["atomic_numbers"] = sample["source_types"]
        sample["symmetry"] = {"number": sample["label"].item()}
        sample["num_points"] = len(sample["atomic_numbers"])
        # clean up keys
        for key in ["coordinates", "source_types", "label"]:
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
