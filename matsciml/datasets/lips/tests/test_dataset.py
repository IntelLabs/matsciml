from __future__ import annotations

import pytest

from matsciml.datasets import lips, transforms
from matsciml.datasets.lips import LiPSDataset, lips_devset


@pytest.mark.dependency()
def test_load_dataset():
    dset = LiPSDataset(lips_devset)
    sample = dset.__getitem__(10)
    assert all([key in sample for key in ["pos", "atomic_numbers", "cell"]])
    assert all([key in sample["targets"] for key in ["energy", "force"]])


@pytest.mark.dependency(depends=["test_load_dataset"])
def test_point_cloud_batch():
    dset = LiPSDataset(lips_devset)
    samples = [dset.__getitem__(index) for index in range(10)]
    batch = dset.collate_fn(samples)
    assert all([key in batch for key in ["pos", "atomic_numbers", "cell"]])
    assert all([key in batch["targets"] for key in ["energy", "force"]])


@pytest.mark.dependency(depends=["test_load_dataset"])
def test_graph_dataset():
    dset = LiPSDataset(
        lips_devset,
        transforms=[transforms.PointCloudToGraphTransform("dgl", cutoff_dist=20.0)],
    )
    sample = dset.__getitem__(10)
    assert "graph" in sample


@pytest.mark.dependency(depends=["test_graph_dataset"])
def test_graph_batch():
    dset = LiPSDataset(
        lips_devset,
        transforms=[transforms.PointCloudToGraphTransform("dgl", cutoff_dist=20.0)],
    )
    samples = [dset.__getitem__(index) for index in range(10)]
    batch = dset.collate_fn(samples)
    assert "graph" in batch
    assert batch["graph"].batch_size == 10


@pytest.mark.dependency(depends=["test_load_dataset"])
def test_dataset_target_keys():
    dset = LiPSDataset(lips_devset)
    assert dset.target_keys == {"regression": ["energy", "force"]}
