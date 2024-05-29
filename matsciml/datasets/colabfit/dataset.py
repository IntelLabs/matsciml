from __future__ import annotations

from pathlib import Path

import torch

from matsciml.common.registry import registry
from matsciml.common.types import BatchDict, DataDict
from matsciml.datasets.base import PointCloudDataset
from matsciml.datasets.utils import concatenate_keys, point_cloud_featurization

POSSIBLE_KEYS = [
    "potential-energy",
    "force",
    "stress",
    "formation-energy",
    "free-energy",
    "band-gap",
    "adsorption",
]


@registry.register_dataset("ColabFitDataset")
class ColabFitDataset(PointCloudDataset):
    __devset__ = Path(__file__).parents[0].joinpath("devset")

    def data_from_key(
        self,
        lmdb_index: int,
        sub_index: int,
    ) -> DataDict:
        """
        Retrieve a sample from the LMDB file.

        Currently, possible target keys which may be present are:

            potential-energy
            force
            stress
            formation-energy
            free-energy
            band-gap
            adsorption-energy

        For now we assume that all samples in the dataset
        contain the expected target properties for the task at hand.
        If that is not the case, this is a result of how the
        original developers composed the dataset.

        # TODO: Fallback or warning for when this does occur

        Parameters
        ----------
        lmdb_index : int
            Index corresponding to LMDB environment from which to parse.
        subindex : int
            Index within an LMDB file that maps to a sample.

        Returns
        -------
        DataDict
            A single sample from a ColabFit Dataset
        """
        data = super().data_from_key(lmdb_index, sub_index)
        coords = data["pos"]
        # check to make sure 3D coordinates
        assert coords.size(-1) == 3
        system_size = coords.size(0)
        node_choices = self.choose_dst_nodes(system_size, self.full_pairwise)
        src_nodes, dst_nodes = (
            node_choices["pc_src_nodes"],
            node_choices["pc_dst_nodes"],
        )
        # typecast atomic numbers
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
        # currently only regression targets present
        for key in POSSIBLE_KEYS:
            if key in data:
                data["targets"][key] = data.get(key)
                data["target_types"]["regression"].append(key)
        return data

    @property
    def target_keys(self) -> dict[str, list[str]]:
        return {"regression": POSSIBLE_KEYS}

    @staticmethod
    def collate_fn(batch: list[DataDict]) -> BatchDict:
        # since this class returns point clouds by default, we have to pad
        # the atom-centered point cloud data
        return concatenate_keys(
            batch,
            pad_keys=["pc_features"],
            unpacked_keys=["sizes", "pc_src_nodes", "pc_dst_nodes"],
        )
