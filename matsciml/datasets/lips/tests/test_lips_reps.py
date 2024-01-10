from __future__ import annotations

from itertools import product

import pytest

from matsciml.datasets.lips import LiPSDataset, lips_devset
from matsciml.datasets.transforms import PointCloudToGraphTransform


def test_pairwise_pointcloud():
    dset = LiPSDataset(lips_devset, full_pairwise=True)
    sample = dset.__getitem__(10)
    assert all(
        [
            key in sample
            for key in [
                "pos",
                "pc_features",
                "sizes",
                "src_nodes",
                "dst_nodes",
                "force",
            ]
        ],
    )
    feats = sample.get("pc_features")
    pos = sample.get("pos")
    assert feats.size(0) == feats.size(1)
    assert pos.ndim == 2


def test_sampled_pointcloud():
    dset = LiPSDataset(lips_devset, full_pairwise=False)
    sample = dset.__getitem__(10)
    assert all(
        [
            key in sample
            for key in [
                "pos",
                "pc_features",
                "sizes",
                "src_nodes",
                "dst_nodes",
                "force",
            ]
        ],
    )
    feats = sample.get("pc_features")
    pos = sample.get("pos")
    assert feats.size(0) >= feats.size(1)
    assert pos.ndim == 2


def test_graph_representation():
    dset = LiPSDataset(
        lips_devset,
        full_pairwise=True,
        transforms=[PointCloudToGraphTransform("dgl", cutoff_dist=15.0)],
    )
    sample = dset.__getitem__(10)
    assert "graph" in sample
    graph = sample.get("graph")
    assert all([key in graph.ndata for key in ["pos", "force", "atomic_numbers"]])


@pytest.mark.parametrize("full_pairwise", [True, False])
def test_batching_pointcloud(full_pairwise):
    dset = LiPSDataset(lips_devset, full_pairwise=full_pairwise)
    samples = [dset.__getitem__(i) for i in range(4)]
    batch = dset.collate_fn(samples)
    assert batch["pos"].shape[-1] == 3
    assert batch["pc_features"].ndim == 4


test_matrix = product(["dgl", "pyg"], [True, False])


@pytest.mark.parametrize("backend, full_pairwise", list(test_matrix))
def test_batching_graph(backend, full_pairwise):
    dset = LiPSDataset(
        lips_devset,
        full_pairwise=full_pairwise,
        transforms=[PointCloudToGraphTransform(backend, cutoff_dist=15.0)],
    )
    samples = [dset.__getitem__(i) for i in range(4)]
    batch = dset.collate_fn(samples)
    assert "graph" in batch
    graph = batch.get("graph")
    if backend == "pyg":
        assert len(graph) == 6
    else:
        assert graph.batch_size == 4
