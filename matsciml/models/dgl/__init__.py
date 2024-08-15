# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: MIT License
from __future__ import annotations

from matsciml.common.packages import package_registry

if package_registry["dgl"]:
    from matsciml.models.dgl.dpp import DimeNetPP
    from matsciml.models.dgl.egnn import PLEGNNBackbone
    from matsciml.models.dgl.gaanet import GalaPotential
    from matsciml.models.dgl.gcn import GraphConvModel
    from matsciml.models.dgl.m3gnet import M3GNet
    from matsciml.models.dgl.megnet import MEGNet
    from matsciml.models.dgl.mpnn import MPNN
    from matsciml.models.dgl.schnet_dgl import SchNet
    from matsciml.models.dgl.tensornet import TensorNet
    from matsciml.models.dgl.chgnet import CHGNet

    __all__ = [
        "DimeNetPP",
        "PLEGNNBackbone",
        "GalaPotential",
        "GraphConvModel",
        "M3GNet",
        "MEGNet",
        "MPNN",
        "SchNet",
        "TensorNet",
        "CHGNet",
    ]
