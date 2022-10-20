# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: MIT License

from typing import Callable

import torch, math
import torch.nn as nn
import numpy as np
from ocpmodels.models.base import AbstractEnergyModel
from dgl.nn.pytorch.factory import KNNGraph
import dgl

from .gaanet_model import MLP, MomentumNorm, LayerNorm

import geometric_algebra_attention.pytorch as gala


class GAANetVectorRegressor(AbstractEnergyModel):
    """Learn a model to regress a (geometric) vector from inputs.

    Stacks permutation-invariant layers that manipulate the values stored
    at each vertex, then adds a permutation-invariant, rotation-covariant layer on top.

    """

    def __init__(
        self,
        D_in,
        hidden_dim=64,
        depth=5,
        dilation=2.0,
        residual=True,
        nonlinearities=True,
        merge_fun="mean",
        join_fun="mean",
        invariant_mode="single",
        covariant_mode="full",
        include_normalized_products=False,
        rank=2,
        invar_value_normalization=None,
        value_normalization=None,
        score_normalization=None,
        block_normalization=None,
        equivariant_attention=True,
    ):
        super().__init__()

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
        self.GAANet_kwargs = dict(merge_fun=merge_fun, join_fun=join_fun)

        self.invar_value_normalization = invar_value_normalization
        self.value_normalization = value_normalization
        self.score_normalization = score_normalization
        self.block_normalization = block_normalization

        self.up_project = torch.nn.Linear(D_in, self.hidden_dim)

        self.make_attention_nets()

        self.nonlin_mlps = []
        if self.nonlinearities:
            self.nonlin_mlps = torch.nn.ModuleList(
                [self.make_value_net(self.hidden_dim) for _ in range(self.depth + 1)]
            )

        # self.final_mlp = self.make_value_net(self.hidden_dim)
        self.energy_projection = torch.nn.Linear(3, 1, bias=False)

    def make_attention_nets(self):
        D_in = lambda i: 1 if (i == self.depth and self.rank == 1) else 2
        self.score_nets = torch.nn.ModuleList(
            [self.make_score_net() for _ in range(self.depth + 1)]
        )
        self.value_nets = torch.nn.ModuleList(
            [self.make_value_net(D_in(i)) for i in range(self.depth + 1)]
        )

        self.final_scale_net = self.make_value_net(self.hidden_dim, 1)

        att_nets = []
        for (scnet, vnet) in zip(self.score_nets, self.value_nets[:-1]):
            rank = max(self.rank, 2)
            att_nets.append(
                gala.VectorAttention(
                    self.hidden_dim,
                    scnet,
                    vnet,
                    reduce=False,
                    rank=rank,
                    **self.GAANet_kwargs
                )
            )
        att_nets.append(
            gala.Vector2VectorAttention(
                self.hidden_dim,
                self.score_nets[-1],
                self.value_nets[-1],
                self.final_scale_net,
                reduce=True,
                rank=self.rank,
                **self.GAANet_kwargs
            )
        )
        self.att_nets = torch.nn.ModuleList(att_nets)

    def make_score_net(self):
        big_D = int(self.hidden_dim * self.dilation)
        layers = [
            torch.nn.Linear(self.hidden_dim, big_D),
        ]

        layers.extend(
            [
                torch.nn.ReLU(),
                torch.nn.Linear(big_D, 1),
            ]
        )
        return torch.nn.Sequential(*layers)

    def make_value_net(self, D_in, D_out=None):
        D_out = D_out or self.hidden_dim
        big_D = int(self.hidden_dim * self.dilation)
        layers = [
            torch.nn.Linear(D_in, big_D),
            torch.nn.LayerNorm(big_D),
        ]
        # if self.dropout:
        #     layers.append(torch.nn.Dropout(self.dropout))

        layers.extend(
            [
                torch.nn.ReLU(),
                torch.nn.Linear(big_D, D_out),
            ]
        )
        return torch.nn.Sequential(*layers)

    def forward(
        self,
        inputs: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:

        _out = self._forward(inputs, positions)

        out_tens = torch.mean(_out, axis=1)

        return out_tens

    def _forward(self, inputs, positions):

        positions = positions / 1000

        (r, v) = (positions, inputs)
        r_pos = torch.as_tensor(r)
        v = torch.as_tensor(v)

        last = self.up_project(v)

        for i, attnet in enumerate(self.att_nets):
            residual = last
            last = attnet((r, last))
            if self.nonlinearities and attnet is not self.att_nets[-1]:
                last = self.nonlin_mlps[i](last)
            if self.residual and attnet is not self.att_nets[-1]:
                last = last + residual

        last = self.energy_projection(last)

        return last


class GalaPotential(AbstractEnergyModel):
    """Calculate a potential using geometric algebra attention

    Stacks permutation-covariant attention blocks, then adds a permutation-invariant reduction layer.

    """

    def __init__(
        self,
        D_in,
        hidden_dim=32,
        depth=2,
        dilation=2.0,
        residual=True,
        nonlinearities=True,
        merge_fun="mean",
        join_fun="mean",
        invariant_mode="single",
        rank=2,
        invar_value_normalization=None,
        value_normalization=None,
        score_normalization=None,
        block_normalization=None,
        pc_size=16,
        pc_mini_batch=50,
    ):
        super().__init__()

        self.pc_size = pc_size
        self.pc_mini_batch = pc_mini_batch
        self.D_in = D_in
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.dilation = dilation
        self.residual = residual
        self.nonlinearities = nonlinearities
        self.rank = rank
        self.invariant_mode = invariant_mode
        self.GAANet_kwargs = dict(
            merge_fun=merge_fun, join_fun=join_fun, invariant_mode=invariant_mode
        )

        self.invar_value_normalization = invar_value_normalization
        self.value_normalization = value_normalization
        self.score_normalization = score_normalization
        self.block_normalization = block_normalization

        self.vec2mv = gala.Vector2Multivector()
        self.up_project = torch.nn.Linear(2 * D_in, self.hidden_dim)
        self.final_mlp = self.make_value_net(self.hidden_dim)
        self.energy_projection = torch.nn.Linear(self.hidden_dim, 1, bias=False)

        self.make_attention_nets()

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

    def make_attention_nets(self):
        D_in = lambda i: 1 if (i == self.depth and self.rank == 1) else 2
        self.score_nets = torch.nn.ModuleList([])
        self.value_nets = torch.nn.ModuleList([])
        self.scale_nets = torch.nn.ModuleList([])
        self.eqvar_att_nets = torch.nn.ModuleList([])
        self.invar_att_nets = torch.nn.ModuleList([])

        for i in range(self.depth + 1):
            reduce = i == self.depth
            rank = max(2, self.rank) if not reduce else self.rank

            # rotation-equivariant (multivector-producing) networks
            self.score_nets.append(self.make_score_net())
            self.value_nets.append(
                self.make_value_net(
                    gala.Multivector2MultivectorAttention.get_invariant_dims(
                        self.rank, self.invariant_mode
                    )
                )
            )
            self.scale_nets.append(self.make_score_net())
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
                        self.rank, self.invariant_mode
                    )
                )
            )
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

    def _get_normalization_layers(self, norm, n_dim):
        if not norm:
            return []
        elif norm == "momentum":
            return [MomentumNorm(n_dim)]
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

    def make_value_net(self, D_in, D_out=None, within_network=True):
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
        inputs: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:

        system_size = inputs.shape[0]

        out_tens = torch.zeros(1, 1).to(inputs.device)

        for ii in range(int(math.ceil(system_size / self.pc_mini_batch))):

            low_idx = self.pc_mini_batch * ii
            high_idx = min(self.pc_mini_batch * (ii + 1), system_size)

            mini_batch_input = inputs[low_idx:high_idx]
            mini_batch_positions = positions[low_idx:high_idx]

            _out = self._forward(mini_batch_input, mini_batch_positions)

            out_tens += torch.sum(_out)

        return out_tens

    def _forward(self, inputs, positions):

        (r, v) = (positions, inputs)
        r = torch.as_tensor(r)
        v = torch.as_tensor(v)

        # Rewrite this to have a neighbor list approach - point cloud around each individual point + the catalyst molecule
        # above make up a point that replaces "last_r" and "last"
        # Loop through graph - pull out neighbors for each individual atom , augment each local point cloud with catalyst molecule
        # batch all of these together

        neighbor_rij = r[..., None, :, :] - r[..., :, None, :]
        neighbor_rij = self.vec2mv(neighbor_rij)
        vplus = v[..., None, :, :] + v[..., :, None, :]
        vminus = v[..., None, :, :] - v[..., :, None, :]
        neighbor_vij = torch.cat([vplus, vminus], axis=-1)

        last_r = neighbor_rij
        last = self.up_project(neighbor_vij)

        for i in range(self.depth + 1):
            residual = last
            residual_r = last_r

            last_r = self.eqvar_att_nets[i]((last_r, last))

            torch.cuda.empty_cache()

            last = self.invar_att_nets[i]((last_r, last))

            torch.cuda.empty_cache()

            if self.nonlinearities:
                last = self.nonlin_mlps[i](last)

            if self.residual and i < self.depth:
                last = last + residual

            if self.block_norm_layers:
                last = self.block_norm_layers[i](last)

            torch.cuda.empty_cache()

            if self.residual:
                last_r = last_r + residual_r
            last_r = last_r + neighbor_rij

            torch.cuda.empty_cache()

        last = self.final_mlp(last)
        # Sum over the neighborhood axis needed when doing neighborhood construction
        last = torch.sum(last, -2)
        last = self.energy_projection(last)

        torch.cuda.empty_cache()

        return last
