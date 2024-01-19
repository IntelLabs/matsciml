# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: MIT License
from __future__ import annotations

from typing import Optional

import dgl
import dgl.function as fn
import numpy as np
import sympy as sym
import torch
from torch import nn

from matsciml.models.dgl.dpp import basis_func as bf

"""
Credit for original code: xnuohz; https://github.com/xnuohz/DimeNet-dgl
"""


"""
Layer initialization
"""


@torch.no_grad()
def glorotho_initialization(module: nn.Module, scale: float = 2.0) -> None:
    if isinstance(module, nn.Linear):
        weights = getattr(module, "weight")
        bias = getattr(module, "bias")
        # initialize weights with orthogonal
        torch.nn.init.orthogonal_(weights)
        scale /= weights.size(-2) + weights.size(-1) * weights.var()
        weights.mul_(scale.sqrt())
        if bias is not None:
            torch.nn.init.zeros_(bias)


class Envelope(nn.Module):
    """
    Envelope function that ensures a smooth cutoff
    """

    def __init__(self, exponent: float):
        super().__init__()

        self.p = exponent + 1
        self.a = -(self.p + 1) * (self.p + 2) / 2
        self.b = self.p * (self.p + 2)
        self.c = -self.p * (self.p + 1) / 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Envelope function divided by r
        x_p_0 = x.pow(self.p - 1)
        x_p_1 = x_p_0 * x
        x_p_2 = x_p_1 * x
        env_val = 1 / x + self.a * x_p_0 + self.b * x_p_1 + self.c * x_p_2
        return env_val


class BesselBasisLayer(nn.Module):
    """
    This module takes an input molecular graph, and expands the
    interatomic distances over a Bessel basis.
    """

    def __init__(
        self,
        num_radial: int,
        cutoff: float,
        envelope_exponent: float | None = 5,
    ):
        super().__init__()

        self.cutoff = cutoff
        self.envelope = Envelope(envelope_exponent)
        self.frequencies = nn.Parameter(torch.Tensor(num_radial))
        self.reset_params()

    @torch.no_grad()
    def reset_params(self):
        torch.arange(1, self.frequencies.numel() + 1, out=self.frequencies).mul_(np.pi)

    def forward(self, edge_distances: torch.Tensor) -> torch.Tensor:
        d_scaled = (edge_distances / self.cutoff).unsqueeze(-1)
        d_cutoff = self.envelope(d_scaled)
        return d_cutoff * torch.sin(self.frequencies * d_scaled)


class SphericalBasisLayer(nn.Module):
    def __init__(
        self,
        num_spherical: int,
        num_radial: int,
        cutoff: int,
        envelope_exponent: float | None = 5,
    ):
        super().__init__()

        assert num_radial <= 64
        self.num_radial = num_radial
        self.num_spherical = num_spherical
        self.cutoff = cutoff
        self.envelope = Envelope(envelope_exponent)

        # retrieve formulas
        self.bessel_formulas = bf.bessel_basis(
            num_spherical,
            num_radial,
        )  # x, [num_spherical, num_radial] sympy functions
        self.sph_harm_formulas = bf.real_sph_harm(
            num_spherical,
        )  # theta, [num_spherical, ] sympy functions
        self.sph_funcs = []
        self.bessel_funcs = []

        # convert to torch functions
        x = sym.symbols("x")
        theta = sym.symbols("theta")
        modules = {"sin": torch.sin, "cos": torch.cos}
        for i in range(num_spherical):
            if i == 0:
                first_sph = sym.lambdify(
                    [theta],
                    self.sph_harm_formulas[i][0],
                    modules,
                )(0)
                self.sph_funcs.append(
                    lambda tensor: torch.zeros_like(tensor) + first_sph,
                )
            else:
                self.sph_funcs.append(
                    sym.lambdify([theta], self.sph_harm_formulas[i][0], modules),
                )
            for j in range(num_radial):
                self.bessel_funcs.append(
                    sym.lambdify([x], self.bessel_formulas[i][j], modules),
                )

    def get_bessel_funcs(self):
        return self.bessel_funcs

    def get_sph_funcs(self):
        return self.sph_funcs


class ResidualLayer(nn.Module):
    def __init__(self, units: int, activation: nn.Module | None = None):
        super().__init__()

        if activation is not None and not isinstance(activation, nn.Module):
            activation = activation()

        self.activation = activation
        self.dense_1 = nn.Linear(units, units)
        self.dense_2 = nn.Linear(units, units)

        self.reset_params()

    def reset_params(self):
        self.apply(glorotho_initialization)

    def forward(self, inputs):
        x = self.dense_1(inputs)
        if self.activation is not None:
            x = self.activation(x)
        x = self.dense_2(x)
        if self.activation is not None:
            x = self.activation(x)
        return inputs + x


class EmbeddingBlock(nn.Module):
    def __init__(
        self,
        emb_size: int,
        num_radial: int,
        bessel_funcs: int,
        cutoff: int,
        envelope_exponent: int,
        num_atom_types: int | None = 95,
        activation: nn.Module | None = None,
    ):
        super().__init__()

        self.bessel_funcs = bessel_funcs
        self.cutoff = cutoff
        if activation is not None and not isinstance(activation, nn.Module):
            activation = activation()
        self.activation = activation
        self.envelope = Envelope(envelope_exponent)
        self.embedding = nn.Embedding(num_atom_types, emb_size)
        self.dense_rbf = nn.Linear(num_radial, emb_size)
        self.dense = nn.Linear(emb_size * 3, emb_size)
        self.reset_params()

    def reset_params(self):
        nn.init.uniform_(self.embedding.weight, a=-np.sqrt(3), b=np.sqrt(3))
        self.dense_rbf.apply(glorotho_initialization)
        self.dense.apply(glorotho_initialization)

    def edge_init(self, edges):
        """msg emb init"""
        # m init
        rbf = self.dense_rbf(edges.data["rbf"])
        if self.activation is not None:
            rbf = self.activation(rbf)

        m = torch.cat([edges.src["h"], edges.dst["h"], rbf], dim=-1)
        m = self.dense(m)
        if self.activation is not None:
            m = self.activation(m)

        # rbf_env init
        d_scaled = edges.data["r"] / self.cutoff
        rbf_env = [f(d_scaled) for f in self.bessel_funcs]
        rbf_env = torch.stack(rbf_env, dim=1)

        d_cutoff = self.envelope(d_scaled)
        rbf_env = d_cutoff[:, None] * rbf_env

        return {"m": m, "rbf_env": rbf_env}

    def forward(self, g: dgl.DGLGraph, atom_embeddings: torch.Tensor):
        g.ndata["h"] = atom_embeddings
        g.apply_edges(self.edge_init)
        return g


class InteractionPPBlock(nn.Module):
    def __init__(
        self,
        emb_size: int,
        int_emb_size: int,
        basis_emb_size: int,
        num_radial: int,
        num_spherical: int,
        num_before_skip: int,
        num_after_skip: int,
        activation: nn.Module | None = None,
    ):
        super().__init__()

        self.activation = activation
        if activation is not None and not isinstance(activation, nn.Module):
            self.activation = self.activation()
        # Transformations of Bessel and spherical basis representations
        self.dense_rbf1 = nn.Linear(num_radial, basis_emb_size, bias=False)
        self.dense_rbf2 = nn.Linear(basis_emb_size, emb_size, bias=False)
        self.dense_sbf1 = nn.Linear(
            num_radial * num_spherical,
            basis_emb_size,
            bias=False,
        )
        self.dense_sbf2 = nn.Linear(basis_emb_size, int_emb_size, bias=False)
        # Dense transformations of input messages
        self.dense_ji = nn.Linear(emb_size, emb_size)
        self.dense_kj = nn.Linear(emb_size, emb_size)
        # Embedding projections for interaction triplets
        self.down_projection = nn.Linear(emb_size, int_emb_size, bias=False)
        self.up_projection = nn.Linear(int_emb_size, emb_size, bias=False)
        # Residual layers before skip connection
        self.layers_before_skip = nn.ModuleList(
            [
                ResidualLayer(emb_size, activation=activation)
                for _ in range(num_before_skip)
            ],
        )
        self.final_before_skip = nn.Linear(emb_size, emb_size)
        # Residual layers after skip connection
        self.layers_after_skip = nn.ModuleList(
            [
                ResidualLayer(emb_size, activation=activation)
                for _ in range(num_after_skip)
            ],
        )

        self.reset_params()

    def reset_params(self):
        self.apply(glorotho_initialization)

    def edge_transfer(self, edges):
        # Transform from Bessel basis to dense vector
        rbf = self.dense_rbf1(edges.data["rbf"])
        rbf = self.dense_rbf2(rbf)
        # Initial transformation
        x_ji = self.dense_ji(edges.data["m"])
        x_kj = self.dense_kj(edges.data["m"])
        if self.activation is not None:
            x_ji = self.activation(x_ji)
            x_kj = self.activation(x_kj)

        x_kj = self.down_projection(x_kj * rbf)
        if self.activation is not None:
            x_kj = self.activation(x_kj)
        return {"x_kj": x_kj, "x_ji": x_ji}

    def msg_func(self, edges):
        sbf = self.dense_sbf1(edges.data["sbf"])
        sbf = self.dense_sbf2(sbf)
        x_kj = edges.src["x_kj"] * sbf
        return {"x_kj": x_kj}

    def forward(self, g, l_g):
        g.apply_edges(self.edge_transfer)

        # nodes correspond to edges and edges correspond to nodes in the original graphs
        # node: d, rbf, o, rbf_env, x_kj, x_ji
        for k, v in g.edata.items():
            l_g.ndata[k] = v

        l_g_reverse = dgl.reverse(l_g, copy_edata=True)
        l_g_reverse.update_all(self.msg_func, fn.sum("x_kj", "m_update"))

        g.edata["m_update"] = self.up_projection(l_g_reverse.ndata["m_update"])
        if self.activation is not None:
            g.edata["m_update"] = self.activation(g.edata["m_update"])
        # Transformations before skip connection
        g.edata["m_update"] = g.edata["m_update"] + g.edata["x_ji"]
        for layer in self.layers_before_skip:
            g.edata["m_update"] = layer(g.edata["m_update"])
        g.edata["m_update"] = self.final_before_skip(g.edata["m_update"])
        if self.activation is not None:
            g.edata["m_update"] = self.activation(g.edata["m_update"])

        # Skip connection
        g.edata["m"] = g.edata["m"] + g.edata["m_update"]

        # Transformations after skip connection
        for layer in self.layers_after_skip:
            g.edata["m"] = layer(g.edata["m"])

        return g


class OutputPPBlock(nn.Module):
    def __init__(
        self,
        emb_size: int,
        out_emb_size: int,
        num_radial: int,
        num_dense: int,
        num_targets: int | None = None,
        activation: nn.Module | None = None,
        extensive: bool | None = True,
        encoder_only: bool = True,
    ):
        if num_targets and encoder_only:
            raise ValueError(f"")
        super().__init__()

        if activation is not None and not isinstance(activation, nn.Module):
            activation = activation()
        self.activation = activation
        self.extensive = extensive
        self.dense_rbf = nn.Linear(num_radial, emb_size, bias=False)
        self.up_projection = nn.Linear(emb_size, out_emb_size, bias=False)
        self.dense_layers = nn.ModuleList(
            [nn.Linear(out_emb_size, out_emb_size) for _ in range(num_dense)],
        )
        if not encoder_only:
            self.dense_final = nn.Linear(out_emb_size, num_targets, bias=False)
        self.encoder_only = encoder_only
        self.reset_params()

    def reset_params(self):
        self.apply(glorotho_initialization)

    def forward(self, g):
        with g.local_scope():
            g.edata["tmp"] = g.edata["m"] * self.dense_rbf(g.edata["rbf"])
            g_reverse = dgl.reverse(g, copy_edata=True)
            g_reverse.update_all(fn.copy_e("tmp", "x"), fn.sum("x", "t"))
            g.ndata["t"] = self.up_projection(g_reverse.ndata["t"])

            for layer in self.dense_layers:
                g.ndata["t"] = layer(g.ndata["t"])
                if self.activation is not None:
                    g.ndata["t"] = self.activation(g.ndata["t"])
            if not self.encoder_only:
                g.ndata["t"] = self.dense_final(g.ndata["t"])
            return dgl.readout_nodes(g, "t", op="sum" if self.extensive else "mean")
