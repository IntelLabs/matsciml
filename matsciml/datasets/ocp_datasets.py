# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: MIT License
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import dgl
import torch
from dgl.dataloading import GraphDataLoader
from munch import Munch

from matsciml.common.registry import registry
from matsciml.datasets.base import BaseLMDBDataset


class OpenCatalystDataset(BaseLMDBDataset):
    @property
    def representation(self) -> str:
        return self._representation

    @representation.setter
    def representation(self, value: str) -> None:
        value = value.lower()
        assert value in [
            "graph",
            "point_cloud",
        ], "Supported representations are 'graph' and 'point_cloud'."
        self._representation = value

    @property
    def pad_keys(self) -> list:
        # in the event this i
        return ["pc_features"]

    @property
    def data_loader(self) -> GraphDataLoader:
        return GraphDataLoader


@registry.register_dataset("S2EFDataset")
class S2EFDataset(OpenCatalystDataset):
    __devset__ = Path(__file__).parents[0].joinpath("dev-s2ef-dgl")

    def data_from_key(
        self,
        lmdb_index: int,
        subindex: int,
    ) -> dict[str, torch.Tensor | dgl.DGLGraph]:
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
            # make graph bidirectional
            graph = dgl.to_bidirected(graph, copy_ndata=True)
            output_data["graph"] = graph
        # This is the case for test set data for s2ef with dgl format.
        elif "graph" in data.keys():
            output_data = data
        # tacking on metadata about the task; energy and force regression
        output_data["targets"] = {}
        output_data["target_types"] = {"regression": [], "classification": []}
        output_data["targets"]["energy"] = data.get("y")
        output_data["targets"]["force"] = output_data["graph"].ndata.get("force")
        for key in ["energy", "force"]:
            output_data["target_types"]["regression"].append(key)
        return output_data

    @property
    def target_keys(self) -> dict[str, list[str]]:
        return {"regression": ["energy", "force"]}


@registry.register_dataset("IS2REDataset")
class IS2REDataset(OpenCatalystDataset):
    __devset__ = Path(__file__).parents[0].joinpath("dev-is2re-dgl")

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
        lmdb_root_path: str | Path,
        transforms: list[Callable] | None = None,
    ) -> None:
        super().__init__(lmdb_root_path, transforms)

    def data_from_key(
        self,
        lmdb_index: int,
        subindex: int,
    ) -> dict[str, torch.Tensor | dgl.DGLGraph]:
        data = super().data_from_key(lmdb_index, subindex)
        # make graph bidirectional if it isn't already
        data["graph"] = dgl.to_bidirected(data["graph"], copy_ndata=True)
        # tacking on metadata about the task; energy
        data["targets"] = {}
        data["target_types"] = {"regression": [], "classification": []}
        for suffix in ["init", "relaxed"]:
            data["targets"][f"energy_{suffix}"] = data.get(f"y_{suffix}")
            data["target_types"]["regression"].append(f"energy_{suffix}")
        return data

    @property
    def target_keys(self) -> dict[str, list[str]]:
        return {"regression": ["energy_init", "energy_relaxed"]}
