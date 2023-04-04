# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: MIT License


from typing import Dict, Union, Optional, List, Callable
from pathlib import Path

import torch
import dgl
from munch import Munch

from ocpmodels.datasets.base import DGLDataset


class S2EFDataset(DGLDataset):
    def data_from_key(
        self, lmdb_index: int, subindex: int
    ) -> Dict[str, Union[torch.Tensor, dgl.DGLGraph]]:
        """
        Overrides the `BaseOCPDataset.data_from_key` function, as there are
        some nuances with unpacking the S2EF data regarding `Munch`, and
        maybe DGLGraphs.

        Essentially, we check if the data read in from the LMDB is a Munch
        object, and if so, we construct the DGLGraph and pack it with the
        labels into a dictionary, consistent with the other datasets.

        Parameters
        ----------
        lmdb_index : int
            Index corresponding to the LMDB file
        subindex : int
            Index within an LMDB referring to the specific data point

        Returns
        -------
        Dict[str, Union[torch.Tensor, dgl.DGLGraph]]
            Dictionary with keys: ["graph", "natoms", "y", "sid", "fid", "cell"]
        """
        # read in data as you would normally
        data = super().data_from_key(lmdb_index, subindex)
        # repackage the data if serialized as Munch
        output_data = {}
        # TODO confirm that this is the only case right now
        if isinstance(data, Munch):
            u, v = data.get("edge_index")
            # create a DGL graph
            graph = dgl.graph((u, v), num_nodes=data.get("natoms"))
            for key in [
                "pos",
                "force",
                "tags",
                "fixed",
                "atomic_numbers",
            ]:
                graph.ndata[key] = data.get(key)
            graph.edata["cell_offsets"] = data.get("cell_offsets")
            # loop over labels
            for key in ["natoms", "y", "sid", "fid", "cell"]:
                output_data[key] = data.get(key)
            output_data["graph"] = graph
        # This is the case for test set data for s2ef with dgl format.
        elif 'graph' in data.keys():
            output_data = data
        output_data["dataset"] = self.__class__.__name__
        return output_data


class IS2REDataset(DGLDataset):
    """
    Currently, this class doesn't have anything special implemented,
    but carries on the abstraction so that if there are modifications
    that need to be made, they can be done within this subclass
    in the same way as S2EF.

    The user is still expected to use this dataset for `IS2RE` task,
    although ostensibly even the `DGLDataset` class would work.
    """

    def __init__(
        self,
        lmdb_root_path: Union[str, Path],
        transforms: Optional[List[Callable]] = None,
    ) -> None:
        super().__init__(lmdb_root_path, transforms)

    def data_from_key(
        self, lmdb_index: int, subindex: int
    ) -> Dict[str, Union[torch.Tensor, dgl.DGLGraph]]:
        data = super().data_from_key(lmdb_index, subindex)
        data["dataset"] = self.__class__.__name__
        return data
