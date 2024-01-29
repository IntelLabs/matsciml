from __future__ import annotations

import pytest
import torch

from matsciml.common import package_registry
from matsciml.datasets import IS2REDataset, is2re_devset
from matsciml.datasets.materials_project import (
    MaterialsProjectDataset,
    materialsproject_devset,
)
from matsciml.datasets.transforms import (
    OCPGraphToPointCloudTransform,
    PointCloudToGraphTransform,
)
from matsciml.datasets.utils import concatenate_keys


@pytest.mark.dependency()
def test_collate_mp_pc():
    # uses point clouds
    dset = MaterialsProjectDataset(materialsproject_devset)
    samples = [dset.__getitem__(i) for i in range(4)]
    batch = concatenate_keys(samples, pad_keys=["pc_features", "atomic_numbers"])
    pos = batch["pos"]
    assert pos.ndim == 2
    # batch size of 4 and 3 dimensions for xyz
    assert pos.size(-1) == 3
    assert pos.size(0) == 8
    assert "mask" in batch
    # now try and collate with the class method
    new_batch = dset.collate_fn(samples)
    assert torch.allclose(batch["pos"], new_batch["pos"])
    assert torch.allclose(batch["pc_features"], new_batch["pc_features"])


if package_registry["pyg"]:

    @pytest.mark.dependency(depends=["test_collate_mp_pc"])
    def test_collate_mp_pyg():
        # uses graphs instead
        dset = MaterialsProjectDataset(
            materialsproject_devset,
            transforms=[PointCloudToGraphTransform("pyg")],
        )
        samples = [dset.__getitem__(i) for i in range(4)]
        # no keys needed to be padded
        batch = concatenate_keys(samples)
        assert "graph" in batch
        graph = batch["graph"]
        # check the batch size
        assert graph.num_graphs == 4
        assert all([key in batch for key in ["targets", "target_types"]])


if package_registry["dgl"]:

    @pytest.mark.dependency(depends=["test_collate_mp_pc"])
    def test_collate_mp_dgl():
        # uses graphs instead
        dset = MaterialsProjectDataset(
            materialsproject_devset,
            transforms=[PointCloudToGraphTransform("dgl")],
        )
        samples = [dset.__getitem__(i) for i in range(4)]
        # no keys needed to be padded
        batch = concatenate_keys(samples)
        assert "graph" in batch
        graph = batch["graph"]
        assert graph.batch_size == 4
        assert all([key in batch for key in ["targets", "target_types"]])
        # now try and collate with the class method
        new_batch = dset.collate_fn(samples)
        assert torch.allclose(
            new_batch["graph"].ndata["atomic_numbers"],
            graph.ndata["atomic_numbers"],
        )
        assert torch.allclose(new_batch["graph"].ndata["pos"], graph.ndata["pos"])


@pytest.mark.dependency(depends=["test_collate_is2re_dgl"])
def test_collate_is2re_pc():
    dset = IS2REDataset(is2re_devset)
    samples = [dset.__getitem__(i) for i in range(4)]
    batch = dset.collate_fn(samples)
    assert len(batch["atomic_numbers"]) == 4
    assert all(
        [key in batch for key in ["pos", "pc_features", "targets", "target_types"]],
    )
    assert sorted(batch.keys())
