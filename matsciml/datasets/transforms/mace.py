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
        max_atom_number: int = 1000,
    ) -> None:
        super().__init__()
        self.energy_weights = energy_weights
        self.forces_weights = forces_weights
        self.stress_weights = stress_weights
        self.virials_weights = virials_weights
        self.atom_table = torch.eye(max_atom_number, dtype=torch.long)

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
        # generate one-hot vectors for the atomic numbers
        atomic_numbers = graph["atomic_numbers"]
        graph["onehot_atomic_numbers"] = self.atom_table[atomic_numbers]
        # now work on structural aspects with periodic boundary conditions
        if "pbc" in data:
            pbc = data["pbc"]
        else:
            pbc = getattr(graph, "pbc")
            if pbc is None:
                raise ValueError("No periodic boundary conditions available in data!")
        shifts = []
        edge_index = []
        unit_shifts = []
        # TODO code below needs to be reviewed and made functional
        b_sz = data["ptr"].numel() - 1
        pos_k = data["positions"].reshape(b_sz, -1, 3)
        cell_k = data["cell"].reshape(b_sz, 3, 3)
        for k in range(b_sz):
            pbc_tensor = pbc[k] == 1
            pbc_tuple = (
                pbc_tensor[0].item(),
                pbc_tensor[1].item(),
                pbc_tensor[2].item(),
            )
            # get_neighborhood is missing from implementation
            edge_index_k, shifts_k, unit_shifts_k = get_neighborhood(
                pos_k[k].numpy(),
                cutoff=self.r_max.item(),
                pbc=pbc_tuple,
                cell=cell_k[k],
            )
            shifts += [torch.Tensor(shifts_k)]
            edge_index += [torch.Tensor(edge_index_k).T.to(torch.int64)]
            unit_shifts += [torch.Tensor(unit_shifts_k)]
        # since modifications are done in place, just return the original
        # structure for consistency
        return data
