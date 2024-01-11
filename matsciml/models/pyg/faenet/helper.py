from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch_geometric.nn import radius_graph


def get_pbc_distances(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    cell: torch.Tensor,
    cell_offsets: torch.Tensor,
    neighbors: torch.Tensor,
    return_offsets: bool = False,
    return_rel_pos: bool = False,
) -> dict[str, torch.Tensor]:
    """Compute distances between atoms with periodic boundary conditions

    Args:
        pos (tensor): (N, 3) tensor of atomic positions
        edge_index (tensor): (2, E) tensor of edge indices
        cell (tensor): (3, 3) tensor of cell vectors
        cell_offsets (tensor): (N, 3) tensor of cell offsets
        neighbors (tensor): (N, 3) tensor of neighbor indices
        return_offsets (bool): return the offsets
        return_rel_pos (bool): return the relative positions vectors

    Returns:
        (dict): dictionary with the updated edge_index, atom distances,
            and optionally the offsets and distance vectors.
    """
    rel_pos = pos[edge_index[0]] - pos[edge_index[1]]

    # correct for pbc
    neighbors = neighbors.to(cell.device)
    cell = torch.repeat_interleave(cell, neighbors, dim=0)
    offsets = cell_offsets.float().view(-1, 1, 3).bmm(cell.float()).view(-1, 3)
    rel_pos += offsets

    # compute distances
    distances = rel_pos.norm(dim=-1)

    # redundancy: remove zero distances
    nonzero_idx = torch.arange(len(distances), device=distances.device)[distances != 0]
    edge_index = edge_index[:, nonzero_idx]
    distances = distances[nonzero_idx]

    out = {
        "edge_index": edge_index,
        "distances": distances,
    }

    if return_rel_pos:
        out["rel_pos"] = rel_pos[nonzero_idx]

    if return_offsets:
        out["offsets"] = offsets[nonzero_idx]

    return out


def base_preprocess(
    data,
    cutoff: int = 6.0,
    max_num_neighbors: int = 40,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Preprocess datapoint: create a cutoff graph,
        compute distances and relative positions.

        Args:
        data (data.Data): data object with specific attributes:
                - batch (N): index of the graph to which each atom belongs to in this batch
                - pos (N,3): atom positions
                - atomic_numbers (N): atomic numbers of each atom in the batch
                - edge_index (2,E): edge indices, for all graphs of the batch
            With B is the batch size, N the number of atoms in the batch (across all graphs),
            and E the number of edges in the batch.
            If these attributes are not present, implement your own preprocess function.
        cutoff (int): cutoff radius (in Angstrom)
        max_num_neighbors (int): maximum number of neighbors per node.

    Returns:
        (tuple): atomic_numbers, batch, sparse adjacency matrix, relative positions, distances
    """
    edge_index = radius_graph(
        data.pos,
        r=cutoff,
        batch=data.batch,
        max_num_neighbors=max_num_neighbors,
    )
    rel_pos = data.pos[edge_index[0]] - data.pos[edge_index[1]]
    return (
        data.atomic_numbers.long(),
        data.batch,
        edge_index,
        rel_pos,
        rel_pos.norm(dim=-1),
    )


def pbc_preprocess(
    data,
    cutoff: int = 6.0,
    max_num_neighbors: int = 40,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Preprocess datapoint using periodic boundary conditions
        to improve the existing graph.

    Args:
        data (data.Data): data object with specific attributes. B is the batch size,
            N the number of atoms in the batch (across all graphs), E the number of edges in the batch.
                - batch (N): index of the graph to which each atom belongs to in this batch
                - pos (N,3): atom positions
                - atomic_numbers (N): atomic numbers of each atom in the batch
                - cell (B, 3, 3): unit cell containing each graph, for materials.
                - cell_offsets (E, 3): cell offsets for each edge, for materials
                - neighbors (B): total number of edges inside each graph.
                - edge_index (2,E): edge indices, for all graphs of the batch
            If these attributes are not present, implement your own preprocess function.

    Returns:
        (tuple): atomic_numbers, batch, sparse adjacency matrix, relative positions, distances
    """
    out = get_pbc_distances(
        data.pos,
        data.edge_index,
        data["cell"],
        data.cell_offsets,
        data.neighbors,
        return_rel_pos=True,
    )

    return (
        data.atomic_numbers.long(),
        data.batch,
        out["edge_index"],
        out["rel_pos"],
        out["distances"],
    )


class GaussianSmearing(nn.Module):
    r"""Smears a distance distribution by a Gaussian function."""

    def __init__(self, start: int = 0.0, stop: int = 5.0, num_gaussians: int = 50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))
