from __future__ import annotations

import pytest
import torch
from dgl import DGLGraph

from matsciml.common.registry import registry
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


@pytest.mark.parametrize("dset_name", valid_dsets)
@pytest.mark.parametrize("graph_type", ["pyg", "dgl"])
def test_noisy_graph(dset_name, graph_type):
    """Test the transform on graph types."""
    dset_class = registry.get_dataset_class(dset_name)
    dset = dset_class.from_devset(
        transforms=[
            NoisyPositions(),
            PeriodicPropertiesTransform(6.0, adaptive_cutoff=True),
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
