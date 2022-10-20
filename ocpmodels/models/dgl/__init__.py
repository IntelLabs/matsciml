# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: MIT License

try:
    import dgl

    _has_dgl = True
except ImportError:
    _has_dgl = False


if _has_dgl:
    from ocpmodels.models.dgl.dpp import DimeNetPP
    from ocpmodels.models.dgl.egnn import PLEGNNBackbone
    from ocpmodels.models.dgl.gaanet import GalaPotential, GAANetVectorRegressor
    from ocpmodels.models.dgl.gcn import GraphConvModel
