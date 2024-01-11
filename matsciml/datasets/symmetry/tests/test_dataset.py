from __future__ import annotations

from matsciml.datasets.symmetry import symmetry_devset
from matsciml.datasets.symmetry.dataset import SyntheticPointGroupDataset
from matsciml.datasets.transforms import PointCloudToGraphTransform


def test_devset_init():
    dset = SyntheticPointGroupDataset(symmetry_devset)
    sample = dset.__getitem__(0)
    assert all([key in sample for key in ["pos", "symmetry", "pc_features"]])


def test_devset_collate():
    dset = SyntheticPointGroupDataset(symmetry_devset)
    samples = [dset.__getitem__(i) for i in range(8)]
    batch = dset.collate_fn(samples)
    assert all([key in batch for key in ["pc_features", "pos", "symmetry"]])


def test_devset_collate():
    dset = SyntheticPointGroupDataset.from_devset([PointCloudToGraphTransform("dgl")])
    samples = [dset.__getitem__(i) for i in range(8)]
    batch = dset.collate_fn(samples)
    assert "graph" in batch
    assert all([key in batch["graph"].ndata for key in ["pos", "atomic_numbers"]])
