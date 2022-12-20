# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: MIT License

from ocpmodels.lightning import data_utils
from ocpmodels.datasets import s2ef_devset, is2re_devset, S2EFDataset
from ocpmodels.datasets import transforms as t


def test_s2ef_dgl_datamodule():
    """
    Use the devset to test the datamodule pipeline by
    grabbing a single batch from the train dataloader.
    """
    dgl_mod = data_utils.S2EFDGLDataModule(s2ef_devset, batch_size=8)
    dgl_mod.setup()
    train_dataloader = dgl_mod.train_dataloader()
    batch = next(iter(train_dataloader))
    # unpack batch
    graph = batch.get("graph")
    assert "force" in graph.ndata.keys()
    assert "pos" in graph.ndata.keys()
    assert len(batch.get("y")) == graph.batch_size


def test_easy_s2ef_datamodule():
    dgl_mod = data_utils.S2EFDGLDataModule.from_devset()


def test_is2re_dgl_datamodule():
    dgl_mod = data_utils.IS2REDGLDataModule(is2re_devset, batch_size=8)
    dgl_mod.setup()
    train_dataloader = dgl_mod.train_dataloader()
    batch = next(iter(train_dataloader))
    # unpack batch
    graph = batch.get("graph")
    assert "pos" in graph.ndata.keys()
    assert len(batch.get("y_init")) == graph.batch_size
    assert len(batch.get("y_init")) == len(batch.get("y_relaxed"))


def test_easy_is2re_datamodule():
    dgl_mod = data_utils.IS2REDGLDataModule.from_devset()


def test_transform_datamodule():
    dgl_mod = data_utils.IS2REDGLDataModule(
        is2re_devset, batch_size=8, transforms=[t.PointCloudTransform(False)]
    )
    dgl_mod.setup()
    train_dataloader = dgl_mod.train_dataloader()
    batch = next(iter(train_dataloader))
    assert "pointcloud_mask" in batch.keys()
