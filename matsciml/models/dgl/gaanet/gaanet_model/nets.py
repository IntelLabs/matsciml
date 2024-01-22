# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: MIT License
from __future__ import annotations

from typing import Callable, List, Tuple, Union

import geometric_algebra_attention.pytorch as gala
import torch
import torch.nn as nn

from matsciml.models.dgl.gaanet.gaanet_model.layers import VectorAttention


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
                self.residuals.append(
                    nn.Linear(
                        in_feats,
                        hidden_feats,
                        bias=False,
                    ),
                )

                for _ in range(num_layers - 2):
                    self.residuals.append(nn.Identity())

            if residual_last:
                if hidden_feats == out_feats:
                    self.residuals.append(nn.Identity())
                else:
                    self.residuals.append(
                        nn.Linear(
                            hidden_feats,
                            out_feats,
                            bias=False,
                        ),
                    )
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
        merge_func: str = "mean",
        join_func: str = "mean",
        rank: int = 2,
        invariant_mode: str = "single",
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.activation = activation
        self.activate_last = activate_last
        self.residual = residual
        self.residual_last = residual_last
        self.layers = nn.ModuleList()

        for _ in range(num_layers - 1):
            self.layers.append(
                VectorAttention(
                    in_feats,
                    hidden_feats,
                    merge_func=merge_func,
                    join_func=join_func,
                    rank=rank,
                    invariant_mode=invariant_mode,
                    residual=residual,
                    reduce=False,
                    dropout=dropout,
                ),
            )

        self.layers.append(
            VectorAttention(
                in_feats,
                hidden_feats,
                merge_func=merge_func,
                join_func=join_func,
                rank=rank,
                invariant_mode=invariant_mode,
                residual=residual_last,
                reduce=True,
                dropout=dropout if dropout_last else 0,
            ),
        )

    def forward(
        self,
        inputs: torch.Tensor,
        positions: torch.Tensor,
        get_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
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


class TiedMultivectorAttention(
    gala.Multivector2MultivectorAttention,
    gala.MultivectorAttention,
):
    def __init__(
        self,
        n_dim,
        score_net,
        value_net,
        scale_net,
        reduce=True,
        merge_fun="mean",
        join_fun="mean",
        rank=2,
        invariant_mode="single",
        covariant_mode="partial",
        include_normalized_products=False,
        convex_covariants=False,
        **kwargs,
    ):
        gala.Multivector2MultivectorAttention.__init__(
            self,
            n_dim=n_dim,
            score_net=score_net,
            value_net=value_net,
            scale_net=scale_net,
            reduce=reduce,
            merge_fun=merge_fun,
            join_fun=join_fun,
            rank=rank,
            invariant_mode=invariant_mode,
            covariant_mode=covariant_mode,
            include_normalized_products=include_normalized_products,
            convex_covariants=convex_covariants,
            **kwargs,
        )

        if type(self) == TiedMultivectorAttention:
            self.init()

    def _evaluate(self, inputs, mask=None):
        parsed_inputs = self._parse_inputs(inputs)
        products = self._get_product_summary(parsed_inputs)
        invar_values = self.value_net(products.summary.invariants)

        joined_values = self._join_fun(invar_values, products.values)
        covariants = self._covariants(products.summary.covariants)
        new_invar_values = products.weights * joined_values
        new_covar_values = products.weights * covariants * self.scale_net(joined_values)

        scores = self.score_net(joined_values)
        old_shape = self.math.shape(scores)

        scores = self._mask_scores(scores, products.broadcast_indices, mask)

        attention, invar_output = self._calculate_attention(
            scores,
            new_invar_values,
            old_shape,
        )
        attention, covar_output = self._calculate_attention(
            scores,
            new_covar_values,
            old_shape,
        )
        output = (covar_output, invar_output)
        return self.OutputType(
            attention,
            output,
            products.summary.invariants,
            invar_values,
            new_invar_values,
        )
