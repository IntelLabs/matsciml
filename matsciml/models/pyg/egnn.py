# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: MIT License
from __future__ import annotations

import torch
from einops import reduce
from torch import nn
from torch_geometric.nn import LayerNorm, MessagePassing
from torch_geometric.nn.pool import global_add_pool
from torch_geometric.typing import Size

from matsciml.common.types import AbstractGraph, Embeddings
from matsciml.models.base import AbstractPyGModel

"""
Some inspiration from https://github.com/lucidrains/egnn-pytorch but
implementation was otherwise from scratch by Kelvin Lee, following
Satorras, Hoogeboom, Welling (2022).
"""


class CoordinateNormalization(nn.Module):
    def __init__(self, epsilon: float = 1e-7) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.scale = nn.Parameter(torch.rand(1), requires_grad=True)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        coord_norm = coords.norm(dim=-1, keepdim=True)
        # rescale coordinates by the norm, then rescale by learnable scale
        new_coords = (coords / (coord_norm + self.epsilon)) * self.scale
        return new_coords


class EGNNConv(MessagePassing):
    """
    Implements a single E(n)-GNN convolution layer, or in Satorras _et al._
    referred to as "Equivariant Graph Convolutional Layer" (EGCL).

    One modification to the architecture is the addition of ``LayerNorm``
    in the messages.
    """

    def __init__(
        self,
        node_dim: int,
        hidden_dim: int,
        out_dim: int,
        coord_dim: int | None = None,
        edge_dim: int = 1,
        activation: str = "SiLU",
        num_layers: int = 2,
        norm_coords: bool = True,
        norm_edge_feats: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # default use same dimensionality as the node ones to make things simpler
        if not edge_dim:
            edge_dim = node_dim
        if not coord_dim:
            coord_dim = node_dim
        # two sets of node features for ij, relative distance, and edge features
        self.edge_mlp = self.make_mlp(
            node_dim * 2 + edge_dim,
            hidden_dim,
            out_dim,
            activation,
            num_layers,
        )
        # include layer norm to the messages
        if norm_edge_feats:
            self.edge_norm = LayerNorm(hidden_dim)
        else:
            self.edge_norm = nn.Identity()
        # this transforms embeds coordinates
        self.coord_mlp = self.make_mlp(
            coord_dim,
            hidden_dim,
            coord_dim,
            activation="SiLU",
            bias=False,
        )
        self.edge_projection = nn.Linear(out_dim, 1, bias=False)
        if norm_coords:
            self.coord_norm = CoordinateNormalization()
        else:
            self.coord_norm = nn.Identity()
        self.node_mlp = self.make_mlp(
            out_dim + node_dim,
            hidden_dim,
            out_dim,
            activation="SiLU",
        )

    @staticmethod
    def make_mlp(
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        activation: str | None,
        num_layers: int = 2,
        **kwargs,
    ) -> nn.Sequential:
        if not activation:
            activation = nn.Identity
        else:
            activation = getattr(nn, activation, None)
            if not activation:
                raise NameError(
                    f"Requested activation {activation}, but not found in torch.nn",
                )
        layers = [nn.Linear(in_dim, hidden_dim, **kwargs), activation()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim, **kwargs), activation()])
        layers.append(nn.Linear(hidden_dim, out_dim, **kwargs))
        return nn.Sequential(*layers)

    def message(self, atom_feats_i, atom_feats_j, edge_attr) -> torch.Tensor:
        # coordinate distances already included as edge_attr
        joint = torch.cat([atom_feats_i, atom_feats_j, edge_attr], dim=-1)
        edge_feats = self.edge_mlp(joint)
        return self.edge_norm(edge_feats)

    def propagate(
        self,
        edge_index: torch.Tensor,
        size: Size | None = None,
        **kwargs,
    ):
        size = self._check_input(edge_index, size)
        kwarg_dict = self._collect(self._user_args, edge_index, size, kwargs)
        msg_kwargs = self.inspector.distribute("message", kwarg_dict)
        agg_kwargs = self.inspector.distribute("aggregate", kwarg_dict)
        update_kwargs = self.inspector.distribute("update", kwarg_dict)

        # pull out some of the expected arguments
        coords = kwargs.get("coords")  # shape [N, 3]
        atom_feats = kwargs.get("atom_feats")  # shape [N, node_dim]
        rel_coords = kwargs.get("rel_coords")  # shape [E, 3]

        # eq 3, calculate messages along edges
        msg_ij = self.message(**msg_kwargs)
        edge_weights = self.edge_projection(msg_ij)

        # eq 5, aggregated messages
        hidden_nodes = self.aggregate(msg_ij, **agg_kwargs)

        # eq 4, add weighted sum to coordinates
        num_edges = edge_index.size(1)
        edge_norm_factor = 1 / (num_edges - 1)
        weighted_distances = edge_norm_factor * self.aggregate(
            rel_coords * edge_weights,
            **agg_kwargs,
        )
        # now update the coordinates
        new_coords = self.coord_norm(coords + weighted_distances)

        # eq 6, transform node features
        new_node_feats = self.node_mlp(torch.cat([hidden_nodes, atom_feats], dim=-1))
        return self.update((new_node_feats, new_coords), **update_kwargs)

    def forward(
        self,
        atom_feats: torch.Tensor,
        coords: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # becomes shape [num_edges, 3]
        rel_coords = coords[edge_index[0]] - coords[edge_index[1]]
        # basically sum of squares
        rel_dist = reduce(rel_coords.square(), "edges xyz -> edges ()", "sum")
        # combined_edge_feats = torch.cat([edge_feats, rel_dist], dim=-1)
        new_nodes, new_coords = self.propagate(
            edge_index,
            atom_feats=atom_feats,
            coords=coords,
            rel_coords=rel_coords,
            edge_attr=rel_dist,
        )
        return (new_nodes, new_coords)


class EGNN(AbstractPyGModel):
    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
        num_conv: int = 3,
        num_atom_embedding: int = 100,
        activation: str = "SiLU",
        pool_norm: str | nn.Module | None = nn.LayerNorm,
        **kwargs,
    ) -> None:
        super().__init__(hidden_dim, num_atom_embedding)
        # embeds coordinates as part of EGNN
        self.coord_embedding = nn.Sequential(
            nn.Linear(3, hidden_dim, bias=False),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )
        # generate sequence of EGNN convolution layers
        self.conv_layers = nn.ModuleList(
            [
                EGNNConv(
                    hidden_dim,
                    hidden_dim,
                    hidden_dim,
                    activation=activation,
                    **kwargs,
                )
                for _ in range(num_conv)
            ],
        )
        # apply normalization before projection layer, this is to help
        # mitigate exploding graph features
        if isinstance(str, pool_norm):
            pool_norm = getattr(nn, pool_norm)
        if pool_norm is None:
            pool_norm = nn.Identity()
        else:
            pool_norm = pool_norm(hidden_dim)
        self.output = nn.Sequential(
            pool_norm,
            nn.Linear(hidden_dim, output_dim, bias=False),
        )

    def _forward(
        self,
        graph: AbstractGraph,
        node_feats: torch.Tensor,
        pos: torch.Tensor,
        edge_feats: torch.Tensor | None = None,
        **kwargs,
    ) -> Embeddings:
        # embed coordinates, then lookup embeddings for atoms and bonds
        coords = self.coord_embedding(pos)
        # loop over each graph layer
        for layer in self.conv_layers:
            node_feats, coords = layer(node_feats, coords, edge_feats, graph.edge_index)
        # use size-extensive pooling
        pooled_data = global_add_pool(node_feats, graph.batch)
        embeddings = Embeddings(pooled_data, node_feats)
        return embeddings
