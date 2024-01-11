from __future__ import annotations

import pytest

from matsciml.datasets.materials_project import (
    MaterialsProjectDataset,
    materialsproject_devset,
)
from matsciml.datasets.transforms import PointCloudToGraphTransform


def test_pairwise_pointcloud():
    dset = MaterialsProjectDataset(materialsproject_devset)
    sample = dset.__getitem__(10)
    assert all(
        [
            key in sample
            for key in ["pos", "pc_features", "sizes", "src_nodes", "dst_nodes"]
        ],
    )
    # for a pairwise point cloud sizes should be equal
    feats = sample["pc_features"]
    assert feats.size(0) == feats.size(1)
    assert sample["pos"].ndim == 2


def test_sampled_pointcloud():
    dset = MaterialsProjectDataset(materialsproject_devset, full_pairwise=False)
    sample = dset.__getitem__(10)
    assert all(
        [
            key in sample
            for key in ["pos", "pc_features", "sizes", "src_nodes", "dst_nodes"]
        ],
    )
    # for a pairwise point cloud sizes should be equal
    feats = sample["pc_features"]
    assert feats.size(0) >= feats.size(1)
    assert sample["pos"].ndim == 2


def test_graph_transform():
    dset = MaterialsProjectDataset(
        materialsproject_devset,
        full_pairwise=False,
        transforms=[PointCloudToGraphTransform("dgl", cutoff_dist=20.0)],
    )
    sample = dset.__getitem__(10)
    assert "graph" in sample
    assert all([key in sample["graph"].ndata for key in ["pos", "atomic_numbers"]])
