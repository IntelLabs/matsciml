# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: MIT License
from __future__ import annotations

from typing import Optional

import torch
from dgllife.model import SchNetGNN
from munch import munchify

from matsciml.common.utils import conditional_grad, get_pbc_distances, radius_graph_pbc
from matsciml.models.base import AbstractEnergyModel
from matsciml.models.dgl.egnn.egnn_model.egnn_backbone import get_backbone


class EGNN_Wrap(AbstractEnergyModel):

    """Wrapper around the EGNN Function using get_backbone function from previous version
    Further documentation needed and added with additional progess
    """

    def __init__(
        self,
        num_atoms: int | None,
        bond_feat_dim: int,
        num_targets: int,
        config,
    ):
        super().__init__()

        config_munch = munchify(config)

        self.backbone = get_backbone(config_munch)

    @conditional_grad(torch.enable_grad())
    def _forward(self, graph_data, label_data):
        z = graph_data.ndata["atomic_numbers"].long()

        pos = graph_data.ndata["pos"]

        edge_dists = torch.cdist(pos, pos)

        if self.otf_graph:
            edge_index, cell_offsets, neighbors = radius_graph_pbc(
                data,
                self.cutoff,
                50,
            )
            data.edge_index = edge_index
            data.cell_offsets = cell_offsets
            data.neighbors = neighbors

        energy = super(SchNetDGLWrap, self).forward(graph_data, z, edge_dists)

        return energy

    def forward(self, batch_data):
        graph_data, label_data = batch_data

        if self.regress_forces:
            graph_data.ndata["pos"].requires_grad_(True)
        energy = self._forward(graph_data, label_data)

        if self.regress_forces:
            forces = -1 * (
                torch.autograd.grad(
                    energy,
                    graph_data.ndata["pos"],
                    grad_outputs=torch.ones_like(energy),
                    create_graph=True,
                )[0]
            )
            return energy, forces
        else:
            return energy

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())
