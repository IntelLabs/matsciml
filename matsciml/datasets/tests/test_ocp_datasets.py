# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: MIT License
from __future__ import annotations

import dgl

from matsciml.datasets import is2re_devset, s2ef_devset
from matsciml.datasets.ocp_datasets import IS2REDataset, S2EFDataset


def test_base_s2ef_read():
    """
    This test will try and obtain the first and last elements
    of the dataset, as well as check that you can get the length
    of the dataset.
    """
    # no transforms
    dset = S2EFDataset(s2ef_devset)
    # get the first entry
    data = dset.__getitem__(0)
    assert all([key in data for key in ["graph", "targets", "target_types"]])
    assert isinstance(
        data["graph"],
        dgl.DGLGraph,
    ), f"Expected graph to be DGLGraph, got {type(data['graph'])}"
    assert all([key in data["graph"].ndata for key in ["pos", "force"]])


def test_base_is2re_read():
    """
    This test will try and obtain the first and last elements
    of the dev IS2RE dataset and check its length
    """
    dset = IS2REDataset(is2re_devset)
    # get the first entry
    data = dset.__getitem__(0)
    assert all([key in data for key in ["graph", "targets", "target_types"]])
    assert isinstance(
        data["graph"],
        dgl.DGLGraph,
    ), f"Expected graph to be DGLGraph, got {type(data['graph'])}"
    assert "pos" in data["graph"].ndata
    assert all([key in data["targets"] for key in ["energy_relaxed", "energy_init"]])


def test_is2re_collate():
    """
    This function tests for the ability for an IS2RE dataset
    to be properly batched.
    """
    dset = IS2REDataset(is2re_devset)
    unbatched = [dset.__getitem__(i) for i in range(5)]
    batched = dset.collate_fn(unbatched)
    # check there are 5 graphs
    assert batched["graph"].batch_size == 5
    # check one of the label shapes is correct
    assert batched["targets"]["energy_init"].size(0) == 5


def test_s2ef_collate():
    """
    This function tests for the ability for an S2EF dataset
    to be properly batched.
    """
    dset = S2EFDataset(s2ef_devset)
    unbatched = [dset.__getitem__(i) for i in range(5)]
    batched = dset.collate_fn(unbatched)
    # check there are 5 graphs
    assert batched["graph"].batch_size == 5
    # check one of the label shapes is correct
    assert batched["targets"]["energy"].size(0) == 5
    num_nodes = batched["graph"].num_nodes()
    assert batched["targets"]["force"].shape == (num_nodes, 3)


def test_s2ef_target_keys():
    dset = S2EFDataset(s2ef_devset)
    assert dset.target_keys == {"regression": ["energy", "force"]}


def test_is2re_target_keys():
    dset = IS2REDataset(is2re_devset)
    assert dset.target_keys == {"regression": ["energy_init", "energy_relaxed"]}
