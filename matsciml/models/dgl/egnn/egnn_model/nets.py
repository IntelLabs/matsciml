# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: MIT License
from __future__ import annotations

from typing import Callable, List, Tuple

import dgl
import torch
import torch.nn as nn

from matsciml.models.dgl.egnn.egnn_model.layers import EquiCoordGraphConv, KLinears


class MLP(nn.Module):
    def __init__(
        self,
        dims: list[int],
        activation: Callable[[torch.Tensor], torch.Tensor] = None,
        activate_last: bool = False,
        bias_last: bool = True,
        k_linears: int = 1,
    ) -> nn.Module:
        super().__init__()
        self._depth = len(dims) - 1
        self._k_linears = k_linears
        self._linear_kwargs = {}

        if k_linears == 1:
            self.linear = nn.Linear
        else:
            self.linear = KLinears
            self._linear_kwargs["k"] = k_linears

        self.layers = nn.ModuleList()

        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            if i < self._depth - 1:
                self.layers.append(
                    self.linear(in_dim, out_dim, bias=True, **self._linear_kwargs),
                )

                if activation is not None:
                    self.layers.append(activation)
            else:
                self.layers.append(
                    self.linear(in_dim, out_dim, bias=bias_last, **self._linear_kwargs),
                )

                if activation is not None and activate_last:
                    self.layers.append(activation)

    def __repr__(self):
        dims = []

        for layer in self.layers:
            if isinstance(layer, (nn.Linear, KLinears)):
                dims.append(f"{layer.in_features} \u2192 {layer.out_features}")
            else:
                dims.append(layer.__class__.__name__)

        return f'MLP({", ".join(dims)})'

    @property
    def last_linear(self) -> nn.Linear:
        for layer in reversed(self.layers):
            if isinstance(layer, (nn.Linear, KLinears)):
                return layer

    @property
    def depth(self) -> int:
        return self._depth

    @property
    def k_linears(self) -> int:
        return self._k_linears

    @property
    def in_features(self) -> int:
        return self.layers[0].in_features

    @property
    def out_features(self) -> int:
        for layer in reversed(self.layers):
            if isinstance(layer, nn.Linear, KLinears):
                return layer.out_features

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs

        for layer in self.layers:
            x = layer(x)

        return x


class EGNN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        depth: int,
        feat_dims: list[int],
        message_dims: list[int],
        position_dims: list[int],
        edge_attributes_dim: int,
        activation: Callable[[torch.Tensor], torch.Tensor],
        residual: bool,
        normalize: bool,
        tanh: bool,
        activate_last: bool = False,
        k_linears: int = 1,
        use_attention: bool = False,
        attention_dims: list[int] = None,
        attention_norm: Callable[[torch.Tensor], torch.Tensor] = None,
        num_atoms_embedding: int = 100,
    ) -> nn.Module:
        super().__init__()
        self._k_linears = k_linears
        message_dim = message_dims[-1]

        position_sizes = [message_dim]
        position_sizes.extend(position_dims)
        position_sizes.append(1)

        hidden_sizes = [hidden_dim for _ in range(depth)]
        hidden_sizes.append(out_dim)

        # use an atomic number lookup to grab from the table of embeddings
        self.atom_embedding = nn.Embedding(num_atoms_embedding, hidden_dim)
        self.in_embed = MLP([hidden_dim, hidden_dim], k_linears=k_linears)
        self.layers = nn.ModuleList()

        for i, (in_dim_, out_dim_) in enumerate(zip(hidden_sizes, hidden_sizes[:-1])):
            edge_in_dim = (2 * in_dim_) + 1 + edge_attributes_dim
            edge_sizes = [edge_in_dim]
            edge_sizes.extend(message_dims)

            feat_sizes = [in_dim_ + message_dim]
            feat_sizes.extend(feat_dims)
            feat_sizes.append(out_dim_)

            if use_attention:
                attention_sizes = [message_dim]

                if attention_dims is not None:
                    attention_sizes.extend(attention_dims)

                attention_sizes.append(1)

            edge_func = MLP(
                edge_sizes,
                activation=activation,
                activate_last=True,
                k_linears=k_linears,
            )
            position_func = MLP(
                position_sizes,
                activation=activation,
                bias_last=False,
                k_linears=k_linears,
            )
            feat_func = MLP(feat_sizes, activation=activation, k_linears=k_linears)

            if use_attention:
                if isinstance(attention_norm, nn.Softmax):
                    attention_activation = nn.LeakyReLU(negative_slope=0.2)
                    attention_activate_last = True
                else:
                    attention_activation = activation
                    attention_activate_last = False

                attention_func = nn.Sequential(
                    MLP(
                        attention_sizes,
                        activation=attention_activation,
                        activate_last=attention_activate_last,
                        k_linears=k_linears,
                    ),
                    attention_norm,
                )

            self.layers.append(
                EquiCoordGraphConv(
                    edge_func,
                    position_func,
                    feat_func,
                    attention_func if use_attention else None,
                    residual=residual,
                    normalize=normalize,
                    tanh=tanh,
                ),
            )

            if i < depth - 1 or activate_last:
                self.layers.append(activation)

    @property
    def num_modes(self) -> int:
        return self._k_linears

    def forward(
        self,
        graph: dgl.DGLGraph,
        node_feats: torch.Tensor,
        positions: torch.Tensor,
        edge_attributes: torch.Tensor = None,
    ) -> tuple[torch.Tensor]:
        pos = positions
        edge_attrs = edge_attributes

        if self._k_linears != 1:
            pos = torch.stack([pos for _ in range(self._k_linears)], dim=1)

            if edge_attrs is not None:
                edge_attrs = torch.stack(
                    [edge_attrs for _ in range(self._k_linears)],
                    dim=1,
                )
        embeddings = self.atom_embedding(node_feats)
        feats = self.in_embed(embeddings)

        for layer in self.layers:
            if isinstance(layer, EquiCoordGraphConv):
                feats, pos = layer(graph, feats, pos, edge_attrs)
            else:
                feats = layer(feats)

        return feats, pos
