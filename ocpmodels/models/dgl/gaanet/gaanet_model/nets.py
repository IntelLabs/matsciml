# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: MIT License

from typing import Callable, List, Tuple, Union

import torch
import torch.nn as nn

from .layers import VectorAttention


class MLP(nn.Module):
    def __init__(
        self,
        in_feats: int,
        hidden_feats: int,
        out_feats: int,
        num_layers: int,
        activation: Callable[[torch.Tensor], torch.Tensor],
        activate_last: bool = False,
        residual: bool = False,
        residual_last: bool = False,
        dropout: float = 0,
        dropout_last: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.activation = activation
        self.activate_last = activate_last
        self.residual_last = residual_last
        self.dropout = nn.Dropout(dropout)
        self.dropout_last = dropout_last

        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(in_feats, hidden_feats))

        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_feats, hidden_feats))

        self.layers.append(nn.Linear(hidden_feats, out_feats))

        if residual:
            self.residuals = nn.ModuleList()

            if in_feats == hidden_feats:
                self.residuals.append(nn.Identity())
            else:
                self.residuals.append(nn.Linear(
                    in_feats, hidden_feats, bias=False))

                for _ in range(num_layers - 2):
                    self.residuals.append(nn.Identity())

            if residual_last:
                if hidden_feats == out_feats:
                    self.residuals.append(nn.Identity())
                else:
                    self.residuals.append(nn.Linear(
                        hidden_feats, out_feats, bias=False))
        else:
            self.residuals = None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs

        for i, layer in enumerate(self.layers):
            if self.residuals is not None:
                if i < self.num_layers - 1 or self.residual_last:
                    res = self.residuals[i](x)

            x = layer(x)

            if self.residuals is not None:
                if i < self.num_layers - 1 or self.residual_last:
                    x = x + res

            if i < self.num_layers - 1 or self.activate_last:
                x = self.activation(x)

            if i < self.num_layers - 1 or self.dropout_last:
                x = self.dropout(x)

        return x


class GAANet(nn.Module):
    def __init__(
        self,
        in_feats: int,
        hidden_feats: int,
        num_layers: int,
        activation: Callable[[torch.Tensor], torch.Tensor],
        activate_last: bool = False,
        residual: bool = False,
        residual_last: bool = False,
        dropout: float = 0,
        dropout_last: bool = False,
        merge_func: str = 'mean',
        join_func: str = 'mean',
        rank: int = 2,
        invariant_mode: str = 'single',
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.activation = activation
        self.activate_last = activate_last
        self.residual = residual
        self.residual_last = residual_last
        self.layers = nn.ModuleList()

        for _ in range(num_layers - 1):
            self.layers.append(VectorAttention(
                in_feats,
                hidden_feats,
                merge_func=merge_func,
                join_func=join_func,
                rank=rank,
                invariant_mode=invariant_mode,
                residual=residual,
                reduce=False,
                dropout=dropout,
            ))

        self.layers.append(VectorAttention(
            in_feats,
            hidden_feats,
            merge_func=merge_func,
            join_func=join_func,
            rank=rank,
            invariant_mode=invariant_mode,
            residual=residual_last,
            reduce=True,
            dropout=dropout if dropout_last else 0,
        ))

    def forward(
        self,
        inputs: torch.Tensor,
        positions: torch.Tensor,
        get_attention: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        x = inputs

        if get_attention:
            attentions = []

        for i, layer in enumerate(self.layers):
            x, attention = layer(x, positions)

            if get_attention:
                attentions.append(attention)

            if i < self.num_layers - 1 or self.activate_last:
                x = self.activation(x)

        if get_attention:
            # TODO (Krzysztof): return mean of attentions?

            return x, attentions

        return x
