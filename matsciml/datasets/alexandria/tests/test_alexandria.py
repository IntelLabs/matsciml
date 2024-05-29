from __future__ import annotations

from matsciml.datasets import transforms
from matsciml.datasets import AlexandriaDataset
from matsciml.datasets.alexandria import AlexandriaRequest
from matsciml.datasets.transforms import (
    PointCloudToGraphTransform,
    PeriodicPropertiesTransform,
)


def test_AlexandriaRequest():
    alexandria_request = AlexandriaRequest.devset(AlexandriaDataset.__devset__)
    alexandria_request.download_and_write(n_jobs=1)
    dset = AlexandriaDataset.from_devset()
    # 98 due to 2 single atom structures that were removed
    assert len(dset) == 98


def test_dataset_collate():
    dset = AlexandriaDataset(AlexandriaDataset.__devset__)
    data = [dset.__getitem__(index) for index in range(10)]
    batch = dset.collate_fn(data)
    # check the nuclear coordinates and numbers match what is expected
    assert batch["pos"].shape[-1] == 3
    assert batch["pos"].ndim == 2
    assert len(batch["atomic_numbers"]) == 10


def test_dgl_dataset():
    dset = AlexandriaDataset(
        AlexandriaDataset.__devset__,
        transforms=[
            PeriodicPropertiesTransform(20.0),
            PointCloudToGraphTransform("dgl", cutoff_dist=20.0),
        ],
    )
    for index in range(10):
        data = dset.__getitem__(index)
        assert "graph" in data


def test_dgl_collate():
    dset = AlexandriaDataset(
        AlexandriaDataset.__devset__,
        transforms=[
            PeriodicPropertiesTransform(20.0),
            transforms.PointCloudToGraphTransform("dgl", cutoff_dist=20.0),
        ],
    )
    data = [dset.__getitem__(index) for index in range(10)]
    batch = dset.collate_fn(data)
    assert "graph" in batch
    # should be ten graphs
    assert batch["graph"].batch_size == 10
    assert all([key in batch["graph"].ndata for key in ["pos", "atomic_numbers"]])


def test_dataset_target_keys():
    dset = AlexandriaDataset.from_devset()
    print(dset.target_keys)
    assert dset.target_keys == {
        "classification": [],
        "regression": [
            "energy_total",
            "total_mag",
            "dos_ef",
            "band_gap_ind",
            "e_form",
            "e_above_hull",
        ],
    }


def test_pairwise_pointcloud():
    dset = AlexandriaDataset.from_devset()
    sample = dset.__getitem__(10)
    assert all(
        [
            key in sample
            for key in ["pos", "pc_features", "sizes", "pc_src_nodes", "pc_dst_nodes"]
        ],
    )
    # for a pairwise point cloud sizes should be equal
    feats = sample["pc_features"]
    assert feats.size(0) == feats.size(1)
    assert sample["pos"].ndim == 2


def test_sampled_pointcloud():
    dset = AlexandriaDataset(AlexandriaDataset.__devset__, full_pairwise=False)
    sample = dset.__getitem__(10)
    assert all(
        [
            key in sample
            for key in ["pos", "pc_features", "sizes", "pc_src_nodes", "pc_dst_nodes"]
        ],
    )
    # for a pairwise point cloud sizes should be equal
    feats = sample["pc_features"]
    assert feats.size(0) >= feats.size(1)
    assert sample["pos"].ndim == 2


def test_graph_transform_dgl():
    dset = AlexandriaDataset(
        AlexandriaDataset.__devset__,
        full_pairwise=False,
        transforms=[
            PeriodicPropertiesTransform(20.0),
            PointCloudToGraphTransform("dgl", cutoff_dist=20.0),
        ],
    )
    sample = dset.__getitem__(10)
    assert "graph" in sample
    assert all([key in sample["graph"].ndata for key in ["pos", "atomic_numbers"]])


def test_graph_transform_pyg():
    dset = AlexandriaDataset(
        AlexandriaDataset.__devset__,
        full_pairwise=False,
        transforms=[
            PeriodicPropertiesTransform(20.0),
            PointCloudToGraphTransform("pyg", cutoff_dist=20.0),
        ],
    )
    sample = dset.__getitem__(10)
    assert "graph" in sample
    assert all([key in sample["graph"] for key in ["pos", "atomic_numbers"]])


def test_graph_transform_pyg_full_pairwise():
    dset = AlexandriaDataset(
        AlexandriaDataset.__devset__,
        full_pairwise=True,
        transforms=[
            PeriodicPropertiesTransform(20.0),
            PointCloudToGraphTransform("pyg", cutoff_dist=20.0),
        ],
    )
    sample = dset.__getitem__(10)
    assert "graph" in sample
    assert all([key in sample["graph"] for key in ["pos", "atomic_numbers"]])
