from __future__ import annotations

import pytest

from matsciml.datasets import transforms
from matsciml.datasets.colabfit import ColabFitDataset


@pytest.mark.dependency()
def test_devset_init():
    """Test whether or not the dataset can be created from devset"""
    dset = ColabFitDataset.from_devset()


@pytest.mark.dependency(depends=["test_devset_init"])
def test_devset_read():
    """Ensure we can read every entry in the devset"""
    dset = ColabFitDataset.from_devset()
    num_samples = len(dset)
    for index in range(num_samples):
        sample = dset.__getitem__(index)


@pytest.mark.dependency(depends=["test_devset_read"])
def test_devset_keys():
    """Ensure the devset contains keys and structure we expect"""
    dset = ColabFitDataset.from_devset()
    sample = dset.__getitem__(50)
    print(sample)
    for key in ["pos", "atomic_numbers", "cell", "pbc"]:
        assert key in sample
    # we know this dataset has regression data
    assert "regression" in sample["target_types"]
    for key in ["potential-energy", "force", "stress"]:
        assert key in sample["targets"]
        assert key in sample["target_types"]["regression"]


@pytest.mark.dependency(depends=["test_devset_read"])
def test_point_cloud_batch():
    dset = ColabFitDataset.from_devset()
    samples = [dset.__getitem__(index) for index in range(10)]
    batch = dset.collate_fn(samples)
    assert all([key in batch for key in ["pos", "atomic_numbers", "cell", "pbc"]])
    assert all(
        [key in batch["targets"] for key in ["potential-energy", "force", "stress"]],
    )


@pytest.mark.dependency(depends=["test_devset_read"])
def test_graph_dataset():
    dset = ColabFitDataset.from_devset(
        transforms=[transforms.PointCloudToGraphTransform("dgl", cutoff_dist=20.0)],
    )
    sample = dset.__getitem__(10)
    assert "graph" in sample


@pytest.mark.dependency(depends=["test_graph_dataset"])
def test_graph_batch():
    dset = ColabFitDataset.from_devset(
        transforms=[transforms.PointCloudToGraphTransform("dgl", cutoff_dist=20.0)],
    )
    samples = [dset.__getitem__(index) for index in range(10)]
    batch = dset.collate_fn(samples)
    assert "graph" in batch
    assert batch["graph"].batch_size == 10


@pytest.mark.dependency(depends=["test_devset_read"])
def test_pairwise_pointcloud():
    dset = ColabFitDataset.from_devset()
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


@pytest.mark.dependency(depends=["test_devset_read"])
def test_sampled_pointcloud():
    dset = ColabFitDataset.from_devset(full_pairwise=False)
    sample = dset.__getitem__(10)
    assert all(
        [
            key in sample
            for key in ["pos", "pc_features", "sizes", "src_nodes", "dst_nodes"]
        ],
    )
    # for a non-pairwise point cloud sizes should not be equal
    feats = sample["pc_features"]
    assert feats.size(0) >= feats.size(1)
    assert sample["pos"].ndim == 2
