from __future__ import annotations

import pytest
import lightning.pytorch as pl
import e3nn
import torch
import mace

from matsciml.datasets import MaterialsProjectDataset
from matsciml.datasets.transforms import (
    PointCloudToGraphTransform,
    PeriodicPropertiesTransform,
    NoisyPositions,
    FrameAveraging,
)
from matsciml.lightning.data_utils import MatSciMLDataModule
from matsciml.models import PLEGNNBackbone, FAENet, MACEWrapper
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


@pytest.fixture
def mace_config():
    model_args = {
        "r_max": 6.0,
        "radial_type": "bessel",
        "distance_transform": None,
        "num_polynomial_cutoff": 5.0,
        "num_interactions": 2,
        "num_bessel": 8,
        "num_atom_embedding": 100,
        "max_ell": 3,
        "gate": torch.nn.SiLU(),
        "interaction_cls": mace.modules.blocks.RealAgnosticResidualInteractionBlock,
        "interaction_cls_first": mace.modules.blocks.RealAgnosticResidualInteractionBlock,
        "correlation": 3,
        "avg_num_neighbors": 31.0,
        "atomic_inter_scale": 0.21,
        "atomic_inter_shift": 0.0,
        "atom_embedding_dim": 128,
        "MLP_irreps": e3nn.o3.Irreps("16x0e"),
        "hidden_irreps": e3nn.o3.Irreps("128x0e + 128x1o"),
        "mace_module": mace.modules.ScaleShiftMACE,
        "disable_forces": False,
    }
    return {"encoder_class": MACEWrapper, "encoder_kwargs": model_args}


def test_force_regression(egnn_config):
    devset = MatSciMLDataModule.from_devset(
        "S2EFDataset",
        dset_kwargs={
            "transforms": [
                PeriodicPropertiesTransform(cutoff_radius=6.0, adaptive_cutoff=True),
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


def test_force_regression_byo_output(mace_config):
    devset = MatSciMLDataModule.from_devset(
        "S2EFDataset",
        dset_kwargs={
            "transforms": [
                PeriodicPropertiesTransform(cutoff_radius=6.0, adaptive_cutoff=True),
                PointCloudToGraphTransform(
                    "pyg",
                    node_keys=["pos", "atomic_numbers"],
                ),
            ],
        },
    )
    task = ForceRegressionTask(**mace_config)
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


def test_force_regression_with_stress(egnn_config):
    egnn_config["compute_stress"] = True
    devset = MatSciMLDataModule.from_devset(
        "S2EFDataset",
        dset_kwargs={
            "transforms": [
                PeriodicPropertiesTransform(cutoff_radius=6.0, adaptive_cutoff=True),
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

    sample = next(iter(devset.train_dataloader()))
    output = task(sample)
    batch_size = devset.train_dataloader().batch_size
    for key in ["energy", "force", "stress", "virials"]:
        assert key in output.keys()
        if key in ["stress", "virials"]:
            assert output[key].shape == (batch_size, 3, 3)


def test_fa_force_regression_with_stress(faenet_config):
    faenet_config["compute_stress"] = True
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

    sample = next(iter(devset.train_dataloader()))
    output = task(sample)
    batch_size = devset.train_dataloader().batch_size
    for key in ["energy", "force", "stress", "virials"]:
        assert key in output.keys()
        if key in ["stress", "virials"]:
            assert output[key].shape == (batch_size, 3, 3)


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
