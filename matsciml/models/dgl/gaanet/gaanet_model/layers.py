# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: MIT License
from __future__ import annotations

from typing import Tuple, Union

import torch
import torch.nn as nn

from matsciml.models.dgl.gaanet.gaanet_model.geometric_algebra import (
    vector_vector,
    vector_vector_invariants,
)

# class MomentumNorm(torch.nn.Module):
#     def __init__(self, n_dim, momentum=.99):
#
#     def forward(self, x):
#
#
#
#
#         if self.training:
#
#
#
#         del sigma
#
#


class MomentumNorm(torch.nn.Module):
    """Exponential decay normalization.

    Computes the mean and standard deviation all axes but the last and
    normalizes values to have mean 0 and variance 1; suitable for
    normalizing a vector of real-valued quantities with differing
    units.

    :param n_dim: Last dimension of the layer input
    :param momentum: Momentum of moving average, from 0 to 1

    """

    def __init__(self, n_dim, momentum=0.99):
        super().__init__()
        self.n_dim = n_dim
        self.register_buffer("momentum", torch.as_tensor(momentum))
        self.register_buffer("mu", torch.zeros(n_dim))
        self.register_buffer("sigma", torch.ones(n_dim))

    def forward(self, x):
        if self.training:
            axes = tuple(range(x.ndim - 1))
            mu_calc = torch.mean(x, axes, keepdim=False)
            sigma_calc = torch.std(x, axes, keepdim=False, unbiased=False)

            new_mu = self.momentum * self.mu + (1 - self.momentum) * mu_calc
            new_sigma = self.momentum * self.sigma + (1 - self.momentum) * sigma_calc

            self.mu[:] = new_mu.detach()
            self.sigma[:] = new_sigma.detach()

        sigma = torch.maximum(self.sigma, torch.as_tensor(1e-7).to(self.sigma.device))

        return (x - self.mu.detach()) / sigma.detach()


class LayerNorm(torch.nn.Module):
    def forward(self, x):
        mu = torch.mean(x, -1, keepdim=True)
        sigmasq = torch.var(x, -1, keepdim=True)

        sigma = torch.sqrt(
            torch.maximum(
                sigmasq,
                (torch.ones(sigmasq.shape) * torch.as_tensor(1e-6)).to(sigmasq.device),
            ),
        )

        ret = (x - mu.detach()) / sigma.detach()

        del sigma
        torch.cuda.empty_cache()

        return ret


class ValueNet(nn.Module):
    def __init__(
        self,
        in_feats: int,
        hidden_feats: int,
        out_feats: int,
        dropout: float = 0,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(in_feats, hidden_feats))
        self.layers.append(nn.LayerNorm(hidden_feats))

        if dropout > 0:
            self.layers.append(nn.Dropout(dropout))

        self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(hidden_feats, out_feats))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs

        for layer in self.layers:
            x = layer(x)

        return x


class ScoreNet(nn.Module):
    def __init__(
        self,
        in_feats: int,
        hidden_feats: int,
        dropout: float = 0,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(in_feats, hidden_feats))

        if dropout > 0:
            self.layers.append(nn.Dropout(dropout))

        self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(hidden_feats, 1))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs

        for layer in self.layers:
            x = layer(x)

        return x


class VectorAttention(nn.Module):
    def __init__(
        self,
        in_feats: int,
        hidden_feats: int,
        merge_func: str = "mean",
        join_func: str = "mean",
        rank: int = 2,  # 2 pairwise attention, 3 triplewise...
        invariant_mode: str = "single",
        residual: bool = False,
        reduce: bool = False,  # True -> permutation invariant
        dropout: float = 0,
    ) -> None:
        super().__init__()
        self.merge_func = merge_func
        self.join_func = join_func
        self.rank = rank
        self.invariant_mode = invariant_mode
        self.residual = residual
        self.reduce = reduce

        self.value_net = ValueNet(
            self.invariant_dims,
            hidden_feats,
            in_feats,
            dropout=dropout,
        )
        self.score_net = ScoreNet(in_feats, hidden_feats, dropout=dropout)

        if merge_func == "concat":
            self.merge_net_left = nn.Linear(in_feats, in_feats, bias=False)
            self.merge_net_right = nn.Linear(in_feats, in_feats, bias=False)

        if join_func == "concat":
            self.join_net_left = nn.Linear(in_feats, in_feats, bias=False)
            self.join_net_right = nn.Linear(in_feats, in_feats, bias=False)

    @property
    def invariant_dims(self) -> int:
        if self.invariant_mode == "full":
            return self.rank**2
        elif self.invariant_mode == "partial":
            return 2 * self.rank - 1
        elif self.invariant_mode == "single":
            return 1 if self.rank == 1 else 2

    def _calculate_invariants(self, positions: torch.Tensor) -> tuple[torch.Tensor]:
        # !! it calculates products only for invarant_mode == 'single' and rank == 2 !!
        # product funcs: algebra.vector_vector, cycle[algebra.bivector_vector, algebra.trivector_vector]
        # invariant funcs: algebra.custom_norm, algebra.vector_vector_invariants,

        # this step could be done in preprocessing (potentially without batching on CPU)

        positions_left = positions.unsqueeze(-3)
        positions_right = positions.unsqueeze(-2)

        product = vector_vector(positions_left, positions_right)
        invariants = vector_vector_invariants(product)

        return invariants

    def _merge(self, inputs: torch.Tensor) -> torch.Tensor:
        left = inputs.unsqueeze(-3)
        right = inputs.unsqueeze(-2)

        if self.merge_func == "mean":
            merged_values = (left + right) / 2
        elif self.merge_func == "concat":
            merged_values_left = self.merge_net_left(left)
            merged_values_right = self.merge_net_right(right)

            merged_values = merged_values_left + merged_values_right

        return merged_values

    def _join(
        self,
        invariant_values: torch.Tensor,
        merged_values: torch.Tensor,
    ) -> torch.Tensor:
        if self.join_func == "mean":
            joined_values = (invariant_values + merged_values) / 2
        elif self.join_func == "concat":
            joined_values_left = self.join_net_left(invariant_values)
            joined_values_right = self.join_net_right(merged_values)

            joined_values = joined_values_left + joined_values_right

        return joined_values

    def _calculate_attention(
        self,
        scores: torch.Tensor,
        inputs: torch.Tensor,
        residual: torch.Tensor = None,
    ) -> torch.Tensor:
        reduce_dims = tuple(-i - 2 for i in range(self.rank - 1))

        attention = torch.softmax(
            scores.flatten(scores.dim() - 3),
            dim=-1,
        ).view(scores.shape)

        x = torch.sum(attention * inputs, dim=reduce_dims)

        if residual is not None:
            x = x + residual

        if self.reduce:
            x = torch.sum(x, dim=-2)

        return x, attention

    def forward(
        self,
        inputs: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor]:
        res = inputs if self.residual else None

        invariants = self._calculate_invariants(positions)

        invariant_values = self.value_net(invariants)
        merged_values = self._merge(inputs)

        joined_values = self._join(invariant_values, merged_values)

        scores = self.score_net(joined_values)
        attention = torch.softmax(self.score_net(joined_values), dim=-1)

        x, attention = self._calculate_attention(scores, joined_values, res)

        return x, attention
