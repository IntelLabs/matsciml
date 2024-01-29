# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: MIT License
from __future__ import annotations

from pathlib import Path
from typing import Callable

import dgl
import torch

from matsciml.common.registry import registry
from matsciml.datasets.base import PointCloudDataset
from matsciml.datasets.utils import point_cloud_featurization


@registry.register_dataset("S2EFDataset")
class S2EFDataset(PointCloudDataset):
    __devset__ = Path(__file__).parents[0].joinpath("dev-s2ef")

    def data_from_key(
        self,
        lmdb_index: int,
        subindex: int,
    ) -> dict[str, torch.Tensor]:
        """
        Overrides the `BaseLMDBDataset.data_from_key` function.

        Essentially, we add in extra point cloud based labels into the
        data dictionary, consistent with the other datasets.

        Parameters
        ----------
        lmdb_index : int
            Index corresponding to the LMDB file
        subindex : int
            Index within an LMDB referring to the specific data point

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary with keys: ["natoms", "y", "sid", "fid", "cell"]
        """
        # read in data as you would normally
        data = super().data_from_key(lmdb_index, subindex)
        system_size = data["pos"].size(0)
        node_choices = self.choose_dst_nodes(system_size, self.full_pairwise)
        src_nodes, dst_nodes = node_choices["src_nodes"], node_choices["dst_nodes"]
        atom_numbers = data["atomic_numbers"].to(torch.int)
        # uses one-hot encoding featurization
        pc_features = point_cloud_featurization(
            atom_numbers[src_nodes],
            atom_numbers[dst_nodes],
            100,
        )
        data["pc_features"] = pc_features
        data["targets"] = {}
        data["target_types"] = {"regression": [], "classification": []}
        data["targets"]["energy"] = data.get("y")
        data["targets"]["force"] = data.get("force")
        for key in ["energy", "force"]:
            data["target_types"]["regression"].append(key)
        return data

    @property
    def target_keys(self) -> dict[str, list[str]]:
        return {"regression": ["energy", "force"]}


@registry.register_dataset("IS2REDataset")
class IS2REDataset(PointCloudDataset):
    __devset__ = Path(__file__).parents[0].joinpath("dev-is2re")

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
    ) -> dict[str, torch.Tensor]:
        data = super().data_from_key(lmdb_index, subindex)
        system_size = data["pos"].size(0)
        node_choices = self.choose_dst_nodes(system_size, self.full_pairwise)
        src_nodes, dst_nodes = node_choices["src_nodes"], node_choices["dst_nodes"]
        atom_numbers = data["atomic_numbers"].to(torch.int)
        # uses one-hot encoding featurization
        pc_features = point_cloud_featurization(
            atom_numbers[src_nodes],
            atom_numbers[dst_nodes],
            100,
        )
        data["pc_features"] = pc_features
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
