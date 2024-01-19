from __future__ import annotations

import pytest
import torch

from matsciml.datasets import S2EFDataset, s2ef_devset, transforms
from matsciml.lightning.data_utils import MatSciMLDataModule


@pytest.mark.dependency()
def test_distance_transform():
    trans = [
        transforms.DistancesTransform(),
    ]
    dset = S2EFDataset(s2ef_devset, transforms=trans)
    batch = dset.__getitem__(0)
    assert "r" in batch.get("graph").edata


@pytest.mark.dependency(["test_distance_transform"])
def test_graph_variable_transform():
    trans = [
        transforms.DistancesTransform(),
        transforms.GraphVariablesTransform(),
    ]
    dset = S2EFDataset(s2ef_devset, transforms=trans)
    batch = dset.__getitem__(0)
    assert "graph_variables" in batch


@pytest.mark.dependency(["test_graph_variable_transform"])
def test_batched_gv_transform():
    trans = [
        transforms.DistancesTransform(),
        transforms.GraphVariablesTransform(),
    ]
    dm = MatSciMLDataModule.from_devset(
        "S2EFDataset",
        dset_kwargs={"transforms": trans},
    )
    dm.setup()
    loader = dm.train_dataloader()
    batch = next(iter(loader))
    assert "graph_variables" in batch
    gv = batch.get("graph_variables")
    assert gv.ndim == 2
    assert gv.shape == (8, 9)
    assert torch.all(~torch.isnan(gv))


@pytest.mark.dependency()
def test_remove_tag_zero():
    trans = [
        transforms.RemoveTagZeroNodes(),
    ]
    dm = MatSciMLDataModule.from_devset(
        "S2EFDataset",
        dset_kwargs={"transforms": trans},
    )
    dm.setup()
    loader = dm.train_dataloader()
    graph = next(iter(loader))["graph"]
    # make sure we've purged all of the tag zero nodes
    assert not torch.any(graph.ndata["tags"] == 0)


@pytest.mark.dependency(["test_remove_tag_zero"])
def test_graph_supernode():
    trans = [
        transforms.GraphSuperNodes(100),
        transforms.RemoveTagZeroNodes(),
    ]
    dm = MatSciMLDataModule.from_devset(
        "S2EFDataset",
        dset_kwargs={"transforms": trans},
    )
    dm.setup()
    loader = dm.train_dataloader()
    graph = next(iter(loader))["graph"]
    # should be one super node per graph
    assert (graph.ndata["tags"] == 3).sum() == graph.batch_size


@pytest.mark.dependency(["test_remove_tag_zero"])
def test_atom_supernode():
    trans = [
        transforms.AtomicSuperNodes(100),
        transforms.RemoveTagZeroNodes(),
    ]
    dm = MatSciMLDataModule.from_devset(
        "S2EFDataset",
        dset_kwargs={"transforms": trans},
    )
    dm.setup()
    loader = dm.train_dataloader()
    graph = next(iter(loader))["graph"]
    # make sure node numbers don't exceed the expected limit
    assert torch.all(graph.ndata["atomic_numbers"] <= 199)
    # make sure we have atomic super nodes after the transform
    assert torch.any(graph.ndata["tags"] == 4)


@pytest.mark.dependency(["test_atom_supernode", "test_graph_supernode"])
def test_all_supernodes():
    trans = [
        transforms.GraphSuperNodes(100),
        transforms.AtomicSuperNodes(100),
        transforms.RemoveTagZeroNodes(),
    ]
    dm = MatSciMLDataModule.from_devset(
        "S2EFDataset",
        dset_kwargs={"transforms": trans},
    )
    dm.setup()
    loader = dm.train_dataloader()
    graph = next(iter(loader))["graph"]
    # make sure node numbers don't exceed the expected limit
    assert torch.all(graph.ndata["atomic_numbers"] <= 200)
    # make sure we have graph and atomic super nodes after the transform
    assert torch.any(graph.ndata["tags"] == 4)
    assert torch.any(graph.ndata["tags"] == 3)
    # check no tag zero nodes remain
    assert not torch.any(graph.ndata["tags"] == 0)
    # make sure the graph super node has an embedding index of 100
    mask = graph.ndata["tags"] == 3
    assert (graph.ndata["atomic_numbers"][mask] - 100).sum() == 0


@pytest.mark.skip(reason="Broken test.")
def test_graph_sorting():
    trans = [
        transforms.GraphReordering("metis", k=10),
    ]
    dm = MatSciMLDataModule.from_devset(
        "S2EFDataset",
        dset_kwargs={"transforms": trans},
    )
    dm.setup()
    loader = dm.train_dataloader()
    graph = next(iter(loader))["graph"]
    # not really anything to test, but just make sure it runs :D
