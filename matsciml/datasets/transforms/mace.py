from __future__ import annotations

import torch

from matsciml.common.types import DataDict, PyGGraph
from matsciml.datasets.transforms.base import AbstractDataTransform

__all__ = ["MACEDataTransform"]


class MACEDataTransform(AbstractDataTransform):
    """
    Transform a PyG graph to add keys/data expected by the MACE architecture.

    This transformation grabs a graph from an individual data sample, checks
    that it is a PyG graph, then starts adding attributes to it as needed by
    MACE. Optionally, weights to specific labels can also be provided either
    as floats (which are broadcast) or as tensors.
    """

    def __init__(
        self,
        energy_weights: float = 1.0,
        forces_weights: float = 1.0,
        stress_weights: float | torch.Tensor = 1.0,
        virials_weights: float | torch.Tensor = 1.0,
    ) -> None:
        super().__init__()
        self.energy_weights = energy_weights
        self.forces_weights = forces_weights
        self.stress_weights = stress_weights
        self.virials_weights = virials_weights

    def __call__(self, data: DataDict) -> DataDict:
        if "graph" not in data:
            raise ValueError(
                "Data sample is missing `graph` key;"
                " make sure it has native graphs, or apply a transform.",
            )
        # this should modify graph attributes directly
        graph = data["graph"]
        if not isinstance(graph, PyGGraph):
            raise TypeError(
                "Graph contained in data sample is not from PyG; "
                "MACE implementation currently only works with PyG.",
            )
        num_nodes = graph.num_nodes
        # use appropriately sized tensors in lieu of data
        if "charges" not in graph:
            graph.charges = torch.zeros((num_nodes,))
        if "energy_weight" not in graph:
            graph.energy_weight = self.energy_weights
        if "forces_weight" not in graph:
            graph.forces_weight = self.forces_weights
        if "stress" not in graph:
            graph.stress = torch.ones((3, 3))
        if "stress_weights" not in graph:
            # should be broadcast if it's float, or elementwise mul if tensor
            graph.stress_weights = torch.ones((3, 3)) * self.stress_weights
        if "virials" not in graph:
            graph.virials = torch.ones((3, 3))
        if "virials_weights" not in graph:
            graph.virials_weights = torch.ones((3, 3)) * self.virials_weights
