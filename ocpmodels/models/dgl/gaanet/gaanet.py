# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: MIT License

from typing import Callable, Optional, Dict, Any, Union

import torch, math
import torch.nn as nn
import numpy as np
from ocpmodels.models.base import AbstractEnergyModel
from dgl.nn.pytorch.factory import KNNGraph
import dgl

from .gaanet_model import MLP, MomentumNorm, LayerNorm, TiedMultivectorAttention
import geometric_algebra_attention.pytorch as gala


class GalaPotential(AbstractEnergyModel):
    """Calculate a potential using geometric algebra attention

    Stacks permutation-covariant attention blocks, then adds a permutation-invariant reduction layer.

    """

    def __init__(
        self,
        D_in: int,
        hidden_dim: int = 64,
        depth: int = 5,
        dilation: float = 2.0,
        residual: bool = True,
        nonlinearities: bool = True,
        merge_fun: str = "mean",
        join_fun: str = "mean",
        invariant_mode: str = "single",
        covariant_mode: str = "full",
        include_normalized_products: bool = False,
        rank: int = 2,
        invar_value_normalization: Optional[str] = None,
        eqvar_value_normalization: Optional[str] = None,
        value_normalization: Optional[str] = None,
        score_normalization: Optional[str] = None,
        block_normalization: Optional[str] = None,
        equivariant_attention: bool = True,
        tied_attention: bool = False,
        encoder_only: Optional[bool] = False,
    ) -> None:
        super().__init__()
        self.tied_attention = tied_attention
        self.equivariant_attention = equivariant_attention
        self.D_in = D_in
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.dilation = dilation
        self.residual = residual
        self.nonlinearities = nonlinearities
        self.rank = rank
        self.invariant_mode = invariant_mode
        self.covariant_mode = covariant_mode
        self.include_normalized_products = include_normalized_products
        self.GAANet_kwargs = dict(
            merge_fun=merge_fun,
            join_fun=join_fun,
            invariant_mode=invariant_mode,
            covariant_mode=covariant_mode,
            include_normalized_products=include_normalized_products,
        )

        self.invar_value_normalization = invar_value_normalization
        self.eqvar_value_normalization = eqvar_value_normalization
        self.value_normalization = value_normalization
        self.score_normalization = score_normalization
        self.block_normalization = block_normalization
        self.encoder_only = encoder_only

        self.vec2mv = gala.Vector2Multivector()
        self.up_project = torch.nn.Linear(2 * D_in, self.hidden_dim)
        self.final_mlp = self.make_value_net(self.hidden_dim)
        self.energy_projection = torch.nn.Linear(self.hidden_dim, 1, bias=False)

        self.make_attention_nets()

        self.int_norm = LayerNorm()

        self.nonlin_mlps = []
        if self.nonlinearities:
            self.nonlin_mlps = torch.nn.ModuleList(
                [
                    self.make_value_net(self.hidden_dim, within_network=False)
                    for _ in range(self.depth + 1)
                ]
            )

        self.block_norm_layers = torch.nn.ModuleList([])
        for _ in range(self.depth + 1):
            self.block_norm_layers.extend(
                self._get_normalization_layers(
                    self.block_normalization, self.hidden_dim
                )
            )

        self.eqvar_norm_layers = torch.nn.ModuleList([])
        if self.equivariant_attention:
            for _ in range(self.depth):
                self.eqvar_norm_layers.extend(
                    self._get_normalization_layers(
                        self.eqvar_value_normalization, self.hidden_dim
                    )
                )

    def make_attention_nets(self) -> None:
        D_in = lambda i: 1 if (i == self.depth and self.rank == 1) else 2
        self.score_nets = torch.nn.ModuleList([])
        self.value_nets = torch.nn.ModuleList([])
        self.scale_nets = torch.nn.ModuleList([])
        self.eqvar_att_nets = torch.nn.ModuleList([])
        self.invar_att_nets = torch.nn.ModuleList([])
        if self.tied_attention:
            self.tied_att_nets = torch.nn.ModuleList([])

        for i in range(self.depth + 1):
            reduce = i == self.depth
            rank = max(2, self.rank) if not reduce else self.rank

            if self.equivariant_attention or self.tied_attention:
                # rotation-equivariant (multivector-producing) networks
                self.score_nets.append(self.make_score_net())
                self.value_nets.append(
                    self.make_value_net(
                        gala.Multivector2MultivectorAttention.get_invariant_dims(
                            self.rank,
                            self.invariant_mode,
                            include_normalized_products=self.include_normalized_products,
                        )
                    )
                )
                self.scale_nets.append(self.make_score_net())
                if self.equivariant_attention:
                    self.eqvar_att_nets.append(
                        gala.Multivector2MultivectorAttention(
                            self.hidden_dim,
                            self.score_nets[-1],
                            self.value_nets[-1],
                            self.scale_nets[-1],
                            reduce=False,
                            rank=rank,
                            **self.GAANet_kwargs
                        )
                    )

            # rotation-invariant (node value-producing) networks
            self.score_nets.append(self.make_score_net())
            self.value_nets.append(
                self.make_value_net(
                    gala.MultivectorAttention.get_invariant_dims(
                        self.rank,
                        self.invariant_mode,
                        include_normalized_products=self.include_normalized_products,
                    )
                )
            )
            if self.tied_attention:
                self.tied_att_nets.append(
                    TiedMultivectorAttention(
                        self.hidden_dim,
                        self.score_nets[-1],
                        self.value_nets[-1],
                        self.scale_nets[-1],
                        reduce=reduce,
                        rank=rank,
                        **self.GAANet_kwargs
                    )
                )
            else:
                self.invar_att_nets.append(
                    gala.MultivectorAttention(
                        self.hidden_dim,
                        self.score_nets[-1],
                        self.value_nets[-1],
                        reduce=reduce,
                        rank=rank,
                        **self.GAANet_kwargs
                    )
                )

    def _get_normalization_layers(self, norm: str, n_dim: int) -> torch.nn.Module:
        if not norm:
            return []
        elif norm == "momentum":
            return [MomentumNorm(n_dim)]
        elif norm == "momentum_layer":
            return [gala.MomentumLayerNormalization()]
        elif norm == "layer":
            return [LayerNorm()]
        else:
            raise NotImplementedError(norm)

    def make_score_net(self):
        big_D = int(self.hidden_dim * self.dilation)
        layers = [
            torch.nn.Linear(self.hidden_dim, big_D),
        ]

        layers.extend(self._get_normalization_layers(self.score_normalization, big_D))

        layers.extend(
            [
                torch.nn.SiLU(),
                torch.nn.Linear(big_D, 1),
            ]
        )
        return torch.nn.Sequential(*layers)

    def make_value_net(
        self,
        D_in: int,
        D_out: Optional[Union[int, None]] = None,
        within_network: bool = True,
    ):
        D_out = D_out or self.hidden_dim
        big_D = int(self.hidden_dim * self.dilation)
        layers = []

        if within_network:
            layers.extend(
                self._get_normalization_layers(self.invar_value_normalization, D_in)
            )

        layers.append(torch.nn.Linear(D_in, big_D))

        layers.extend(self._get_normalization_layers(self.value_normalization, big_D))

        layers.extend(
            [
                torch.nn.SiLU(),
                torch.nn.Linear(big_D, D_out),
            ]
        )
        return torch.nn.Sequential(*layers)

    def forward(
        self,
        batch: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        inputs = batch["pc_features"]
        positions = batch["pos"]

        _out = self._forward(inputs, positions)

        out_tens = torch.mean(_out, axis=1)

        return out_tens

    def _forward(self, inputs: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:

        positions = torch.div(positions, 1)

        last_r_mv = self.vec2mv(positions)
        last_r = last_r_mv
        last = self.up_project(inputs)

        for i in range(self.depth + 1):
            residual = last
            residual_r = last_r

            if self.equivariant_attention:
                last_r = self.eqvar_att_nets[i]((last_r, last))

            if self.tied_attention:
                x_last, last = self.tied_att_nets[i]((last_r, last))
            else:
                last = self.invar_att_nets[i]((last_r, last))

            if self.nonlinearities:
                last = self.nonlin_mlps[i](last)

            if self.residual and i < self.depth:
                last = last + residual

            if self.block_norm_layers:
                last = self.block_norm_layers[i](last)

            # Apply Layer Norm (Momentum) to last_r 
            if self.equivariant_attention:
                if self.residual:
                    last_r = last_r + residual_r
                # Normalize with momentum here
                if i < len(self.eqvar_norm_layers):
                    last_r = self.eqvar_norm_layers[i](last_r)

            if self.residual and self.equivariant_attention:
                last_r = last_r + residual_r
                # Normalize with momentum here

        last = self.final_mlp(last)
        # Sum over the neighborhood axis needed when doing neighborhood construction
        # last = torch.sum(last, -2)
        if not self.encoder_only:
            last = self.energy_projection(last)

        return last
