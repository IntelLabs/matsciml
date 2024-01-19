# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: MIT License
from __future__ import annotations

try:
    import dgl

    _has_dgl = True
except ImportError:
    _has_dgl = False

import pytest


@pytest.mark.skipif(not _has_dgl, reason="DGL not installed; skipping DGL model tests.")
def test_dgl_dpp():
    from matsciml.models.dgl import DimeNetPP


@pytest.mark.skipif(not _has_dgl, reason="DGL not installed; skipping DGL model tests.")
def test_dgl_egnn():
    from matsciml.models.dgl import PLEGNNBackbone


@pytest.mark.skipif(not _has_dgl, reason="DGL not installed; skipping DGL model tests.")
def test_dgl_gcn():
    from matsciml.models.dgl import GraphConvModel


@pytest.mark.skipif(not _has_dgl, reason="DGL not installed; skipping DGL model tests.")
def test_dgl_gaanet():
    from matsciml.models.dgl import GalaPotential


@pytest.mark.skipif(not _has_dgl, reason="DGL not installed; skipping DGL model tests.")
def test_dgl_gaanet():
    from matsciml.models.dgl import M3GNet
