from __future__ import annotations

from matsciml.common.types import DataDict
from matsciml.datasets.transforms.base import AbstractDataTransform
from matgl.graph.compute import compute_pair_vector_and_distance
import torch

__all__ = ["MGLDataTransform"]


class MGLDataTransform(AbstractDataTransform):
    """
    Implements a transform to add or swap keys required by matgl models.
    """

    def __call__(self, data: DataDict) -> DataDict:
        assert (
            "offsets" in data
        ), "Offsets missing from data sample! Make sure to include PeriodicPropertiesTransform in the datamodule."
        graph = data["graph"]
        graph.edata["pbc_offset"] = data["images"]
        graph.edata["pbc_offshift"] = graph.edata["offsets"]
        graph.ndata["node_type"] = graph.ndata["atomic_numbers"].type(torch.int)
        bond_vec, bond_dist = compute_pair_vector_and_distance(graph)
        graph.edata["bond_vec"] = bond_vec
        graph.edata["bond_dist"] = bond_dist
        return data
