from __future__ import annotations

import pytest
import torch
from dgl import DGLGraph

from matsciml.common.registry import registry
from matsciml.lightning import MatSciMLDataModule
from matsciml.datasets.transforms.pretraining import NoisyPositions
from matsciml.datasets.transforms import (
    PeriodicPropertiesTransform,
    PointCloudToGraphTransform,
)


dset_names = registry.__entries__["datasets"].keys()
valid_dsets = list(
    filter(
        lambda x: all(
            [match not in x for match in ["PyG", "Multi", "Cdvae", "PointGroup"]]
        ),
        dset_names,
    )
)


@pytest.mark.parametrize("dset_name", valid_dsets)
def test_noisy_pointcloud(dset_name):
    """Test the transform on the raw point clouds"""
    dset_class = registry.get_dataset_class(dset_name)
    dset = dset_class.from_devset(transforms=[NoisyPositions()])
    for index in range(10):
        sample = dset.__getitem__(index)
        assert "noisy_pos" in sample
        assert torch.isfinite(sample["noisy_pos"]).all()
        assert "pretraining" in sample["target_types"]
        assert "denoise" in sample["target_types"]["pretraining"]


@pytest.mark.parametrize("dset_name", valid_dsets)
@pytest.mark.parametrize("graph_type", ["pyg", "dgl"])
def test_noisy_graph(dset_name, graph_type):
    """Test the transform on graph types."""
    dset_class = registry.get_dataset_class(dset_name)
    dset = dset_class.from_devset(
        transforms=[
            PeriodicPropertiesTransform(6.0, adaptive_cutoff=True),
            NoisyPositions(),
            PointCloudToGraphTransform(
                graph_type, node_keys=["atomic_numbers", "pos", "noisy_pos"]
            ),
        ]
    )
    for index in range(10):
        sample = dset.__getitem__(index)
        graph = sample["graph"]
        if isinstance(graph, DGLGraph):
            target = graph.ndata
        else:
            target = graph
        assert "noisy_pos" in target
        assert torch.isfinite(target["noisy_pos"]).all()
        assert "pretraining" in sample["target_types"]
        assert "denoise" in sample["target_types"]["pretraining"]


@pytest.mark.parametrize("dset_name", valid_dsets)
def test_noisy_pointcloud_datamodule(dset_name):
    """Test the transform on point cloud types with batching."""
    dm = MatSciMLDataModule.from_devset(
        dset_name, batch_size=4, dset_kwargs={"transforms": [NoisyPositions()]}
    )
    dm.setup("fit")
    loader = dm.train_dataloader()
    batch = next(iter(loader))
    assert "noisy_pos" in batch
    assert torch.isfinite(batch["noisy_pos"]).all()
    assert "pretraining" in batch["target_types"]
    assert "denoise" in batch["target_types"]["pretraining"]


@pytest.mark.parametrize("dset_name", valid_dsets)
@pytest.mark.parametrize("graph_type", ["pyg", "dgl"])
def test_noisy_graph_datamodule(dset_name, graph_type):
    """Test the transform on graph types with batching."""
    dm = MatSciMLDataModule.from_devset(
        dset_name,
        dset_kwargs=dict(
            transforms=[
                PeriodicPropertiesTransform(6.0, adaptive_cutoff=True),
                NoisyPositions(),
                PointCloudToGraphTransform(
                    graph_type, node_keys=["atomic_numbers", "pos", "noisy_pos"]
                ),
            ],
        ),
        batch_size=4,
    )
    dm.setup("fit")
    loader = dm.train_dataloader()
    batch = next(iter(loader))
    graph = batch["graph"]
    if isinstance(graph, DGLGraph):
        target = graph.ndata
    else:
        target = graph
    assert "noisy_pos" in target
    assert torch.isfinite(target["noisy_pos"]).all()
    assert "pretraining" in batch["target_types"]
    assert "denoise" in batch["target_types"]["pretraining"]
