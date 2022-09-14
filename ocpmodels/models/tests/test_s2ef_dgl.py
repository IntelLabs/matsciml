# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: MIT License

import pytest


import torch
import pytorch_lightning as pl
from torch import nn

from ocpmodels.lightning.data_utils import S2EFDGLDataModule
from ocpmodels.models import base, GraphConvModel

"""
This test suite will test the final components of the pipeline,
going from model instantiation to S2EFLitModule instantiation,
to running a single training iteration, to testing DDP functionality.

The DDP functionality is only triggered if the single instance
is successful. Similarly, the analogous GPU tests will only run
if there are GPUs available and the basic CPU test passes.
"""

# use a single thread as to not use lots of resources
torch.set_num_threads(1)

# instantiate the devset globally; could've used a fixture
# but the purpose here is not to test the devset_module
# TODO use the devset mechanism when it's merged into development
devset_module = S2EFDGLDataModule.from_devset(batch_size=8, num_workers=0)
devset_module.setup()


@pytest.fixture(scope="module")
def test_graphconv_instantiation() -> GraphConvModel:
    """Test if we can create a very basic graph convolution model"""
    # define a bogstandard graph conv neural network to use for tests
    basic_gnn = GraphConvModel(
        128, 32, num_blocks=2, num_fc_layers=2, activation=nn.SiLU
    )
    return basic_gnn


@pytest.fixture(scope="module")
def test_s2ef_instantiation(test_graphconv_instantiation):
    """Test if an `S2EFLitModule` object can be successfully created"""
    model = base.S2EFLitModule(test_graphconv_instantiation, lr=1e-3, gamma=0.1)
    return model


def test_graphconv_forward(test_graphconv_instantiation):
    """Test if the basic DGL GNN model can run a single pass of the data."""
    # devset_module = DGLDataModule.from_devset()
    # devset_module.setup()
    train_loader = devset_module.train_dataloader()
    # grab a single batch
    batch = next(iter(train_loader))
    input_graph = batch.get("graph")
    with torch.no_grad():
        pred_Y = test_graphconv_instantiation(input_graph)


@pytest.mark.dependency()
def test_s2ef_training(test_s2ef_instantiation):
    """Test if a single training pass can be done"""
    # devset_module = DGLDataModule.from_devset()

    trainer = pl.Trainer(
        accelerator="cpu",
        num_sanity_val_steps=0,
        max_steps=1,
        logger=False,
        enable_model_summary=False,
    )

    trainer.fit(test_s2ef_instantiation, datamodule=devset_module)


@pytest.mark.dependency(depends=["test_s2ef_training"])
def test_s2ef_training_ddp(test_s2ef_instantiation):
    """Test if a single training pass can be done"""
    # devset_module = DGLDataModule.from_devset()

    trainer = pl.Trainer(
        strategy="ddp",
        accelerator="cpu",
        num_sanity_val_steps=0,
        max_steps=1,
        logger=False,
        enable_model_summary=False,
        devices=2,
        num_nodes=1,
    )

    trainer.fit(test_s2ef_instantiation, datamodule=devset_module)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="No GPU, not testing GPU training."
)
@pytest.mark.dependency(depends=["test_s2ef_training"])
def test_s2ef_training_gpu(test_s2ef_instantiation):
    """Test if a single training pass can be done on a single GPU"""
    # devset_module = DGLDataModule.from_devset()

    trainer = pl.Trainer(
        accelerator="gpu",
        num_sanity_val_steps=0,
        max_steps=1,
        logger=False,
        enable_model_summary=False,
        devices=1,
    )

    trainer.fit(test_s2ef_instantiation, datamodule=devset_module)


@pytest.mark.skipif(
    not all([torch.cuda.is_available(), torch.cuda.device_count() < 2]),
    reason="No GPU, not testing GPU training.",
)
@pytest.mark.dependency(depends=["test_s2ef_training_gpu"])
def test_s2ef_training_gpu_ddp(test_s2ef_instantiation):
    """
    Test if a single training pass can be done on a two GPUs with DDP.
    This test depends on the previous; if the single GPU case fails
    there's no reason to try DDP!
    """
    # devset_module = DGLDataModule.from_devset()

    trainer = pl.Trainer(
        strategy="ddp",
        accelerator="gpu",
        num_sanity_val_steps=0,
        max_steps=1,
        logger=False,
        enable_model_summary=False,
        devices=2,
        num_nodes=1,
    )

    trainer.fit(test_s2ef_instantiation, datamodule=devset_module)
