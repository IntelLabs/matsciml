from __future__ import annotations

from collections.abc import Iterable
from typing import Dict, List, Union

import dgl
import torch
from torch.utils.data import ConcatDataset

from matsciml.common.registry import registry
from matsciml.datasets.base import BaseLMDBDataset

# quasi-registry of functions for collating based on dataset class name
collate_registry = {
    dset_name: dset_class.collate_fn
    for dset_name, dset_class in registry.__entries__["datasets"].items()
}


@registry.register_dataset("MultiDataset")
class MultiDataset(ConcatDataset):
    """
    Abstraction layer for combining multiple datasets within Open MatSciML Toolkit.

    This class acts to orchestrate various datasets; primary action is
    to behave similarly to a single dataset (i.e. implement `collate_fn`)
    which will batch data according to the corresponding dataset's collate
    function.
    """

    def __init__(self, datasets: Iterable[BaseLMDBDataset]) -> None:
        super().__init__(datasets)

    @staticmethod
    def collate_fn(
        batch: list[dict[str, torch.Tensor | dgl.DGLGraph | dict[str, torch.Tensor]]],
    ) -> dict[str, dict[str, torch.Tensor | dgl.DGLGraph | dict[str, torch.Tensor]]]:
        """
        Collate function for multiple datasets.

        The main utility of this function is to organize data and batch them
        according to their respective datasets. In effect, we can have a batch
        that contains data from Open Catalyst, Materials Project, etc.

        Parameters
        ----------
        batch : List[Dict[str, Union[torch.Tensor, dgl.DGLGraph, Dict[str, torch.Tensor]]]]
            List of samples, where if the `DataLoader` is shuffled, will contain a
            mixture from different datasets.

        Returns
        -------
        Dict[str, Dict[str, Union[torch.Tensor, dgl.DGLGraph, Dict[str, torch.Tensor]]]]
            Batched data nested under respective datasets. This means that
            tasks can map onto their corresponding dataset (hopefully).
        """
        # collate data based on dataset origin
        all_data = {}
        # bin them into their respective datasets
        for entry in batch:
            origin = entry["dataset"]
            if origin not in all_data:
                all_data[origin] = []
            all_data[origin].append(entry)
        # convert the samples into batched data
        for key in all_data.keys():
            all_data[key] = collate_registry[key](all_data[key])
        return all_data

    @property
    def target_keys(self) -> dict[str, dict[str, list[str]]]:
        keys = {}
        for dset in self.datasets:
            name = dset.__class__.__name__
            keys[name] = dset.target_keys
        return keys
