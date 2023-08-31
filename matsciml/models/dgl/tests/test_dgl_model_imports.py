# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: MIT License

try:
    import dgl

    _has_dgl = True
except ImportError:
    _has_dgl = False

import pytest


@pytest.mark.skipif(not _has_dgl, reason="DGL not installed; skipping DGL model tests.")
def test_dgl_dpp():
    from ocpmodels.models.dgl import DimeNetPP


@pytest.mark.skipif(not _has_dgl, reason="DGL not installed; skipping DGL model tests.")
def test_dgl_egnn():
    from ocpmodels.models.dgl import PLEGNNBackbone


@pytest.mark.skipif(not _has_dgl, reason="DGL not installed; skipping DGL model tests.")
def test_dgl_gcn():
    from ocpmodels.models.dgl import GraphConvModel


@pytest.mark.skipif(not _has_dgl, reason="DGL not installed; skipping DGL model tests.")
def test_dgl_gaanet():
    from ocpmodels.models.dgl import GalaPotential
