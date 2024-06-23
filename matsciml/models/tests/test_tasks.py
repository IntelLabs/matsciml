from __future__ import annotations

import pytest
import pytorch_lightning as pl

from matsciml.datasets.materials_project import MaterialsProjectDataset
from matsciml.datasets.transforms import (
    PointCloudToGraphTransform,
    PeriodicPropertiesTransform,
    NoisyPositions,
    FrameAveraging,
)
from matsciml.lightning.data_utils import MatSciMLDataModule
from matsciml.models import PLEGNNBackbone, FAENet
from matsciml.models.base import (
    ForceRegressionTask,
    GradFreeForceRegressionTask,
    NodeDenoisingTask,
)

pl.seed_everything(2156161)


def test_regression_devset():
    dset = MaterialsProjectDataset.from_devset()  # noqa: F841


@pytest.fixture
def egnn_config():
    model_args = {
        "embed_in_dim": 128,
        "embed_hidden_dim": 32,
        "embed_out_dim": 128,
        "embed_depth": 5,
        "embed_feat_dims": [128, 128, 128],
        "embed_message_dims": [128, 128, 128],
        "embed_position_dims": [64, 64],
        "embed_edge_attributes_dim": 0,
        "embed_activation": "relu",
        "embed_residual": True,
        "embed_normalize": True,
        "embed_tanh": True,
        "embed_activate_last": False,
        "embed_k_linears": 1,
        "embed_use_attention": False,
        "embed_attention_norm": "sigmoid",
        "readout": "sum",
        "node_projection_depth": 3,
        "node_projection_hidden_dim": 128,
        "node_projection_activation": "relu",
        "prediction_out_dim": 1,
        "prediction_depth": 3,
        "prediction_hidden_dim": 128,
        "prediction_activation": "relu",
        "encoder_only": True,
    }
    return {"encoder_class": PLEGNNBackbone, "encoder_kwargs": model_args}


@pytest.fixture
def faenet_config():
    model_args = {
        "average_frame_embeddings": False,
        "pred_as_dict": False,
        "hidden_channels": 128,
        "out_dim": 128,
        "tag_hidden_channels": 0,
    }
    return {"encoder_class": FAENet, "encoder_kwargs": model_args}


def test_force_regression(egnn_config):
    devset = MatSciMLDataModule.from_devset(
        "S2EFDataset",
        dset_kwargs={
            "transforms": [
                PointCloudToGraphTransform(
                    "dgl",
                    cutoff_dist=20.0,
                    node_keys=["pos", "atomic_numbers"],
                ),
            ],
        },
    )
    task = ForceRegressionTask(**egnn_config)
    trainer = pl.Trainer(max_steps=5, logger=False, enable_checkpointing=False)
    trainer.fit(task, datamodule=devset)
    # make sure losses are tracked
    for key in ["energy", "force"]:
        assert f"train_{key}" in trainer.logged_metrics


def test_fa_force_regression(faenet_config):
    devset = MatSciMLDataModule.from_devset(
        "S2EFDataset",
        dset_kwargs={
            "transforms": [
                PeriodicPropertiesTransform(6.0, True),
                PointCloudToGraphTransform(
                    "pyg",
                    node_keys=["pos", "force", "atomic_numbers"],
                ),
                FrameAveraging(frame_averaging="3D", fa_method="stochastic"),
            ],
        },
    )
    task = ForceRegressionTask(**faenet_config)
    trainer = pl.Trainer(max_steps=5, logger=False, enable_checkpointing=False)
    trainer.fit(task, datamodule=devset)
    # make sure losses are tracked
    for key in ["energy", "force"]:
        assert f"train_{key}" in trainer.logged_metrics


def test_gradfree_force_regression(egnn_config):
    devset = MatSciMLDataModule.from_devset(
        "S2EFDataset",
        dset_kwargs={
            "transforms": [
                PointCloudToGraphTransform(
                    "dgl",
                    cutoff_dist=20.0,
                    node_keys=["pos", "atomic_numbers"],
                ),
            ],
        },
    )
    task = GradFreeForceRegressionTask(**egnn_config)
    trainer = pl.Trainer(max_steps=5, logger=False, enable_checkpointing=False)
    trainer.fit(task, datamodule=devset)
    # make sure losses are tracked
    assert "train_force" in trainer.logged_metrics


def test_denoising_task(egnn_config):
    """Tests the denoising task on materials project with EGNN"""
    dm = MatSciMLDataModule.from_devset(
        "MaterialsProjectDataset",
        dset_kwargs={
            "transforms": [
                NoisyPositions(),
                PeriodicPropertiesTransform(6.0, True),
                PointCloudToGraphTransform(
                    "dgl", node_keys=["atomic_numbers", "noisy_pos", "force"]
                ),
            ]
        },
        batch_size=8,
    )
    task = NodeDenoisingTask(**egnn_config, task_keys=["denoise"])
    trainer = pl.Trainer(fast_dev_run=5)
    trainer.fit(task, datamodule=dm)
    assert "train_denoise" in trainer.logged_metrics
