from __future__ import annotations

import pytest
import pytorch_lightning as pl

from matsciml.datasets.materials_project import MaterialsProjectDataset
from matsciml.datasets.transforms import (
    PointCloudToGraphTransform,
)
from matsciml.lightning.data_utils import MatSciMLDataModule
from matsciml.models import PLEGNNBackbone
from matsciml.models.base import ForceRegressionTask, ScalarRegressionTask

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
    # Scenario where task_keys are set. Expect to use task_loss_scaling.
    task = ForceRegressionTask(
        **egnn_config,
        task_keys=["energy", "force"],
        task_loss_scaling={"energy": 1, "force": 10},
    )
    trainer = pl.Trainer(max_steps=5, logger=False, enable_checkpointing=False)
    trainer.fit(task, datamodule=devset)
    # make sure losses are tracked
    for key in ["energy", "force"]:
        assert f"train_{key}" in trainer.logged_metrics

    # Scenario where task_keys are not set. Expect to still use task_loss_scaling.
    task = ForceRegressionTask(
        **egnn_config,
        task_loss_scaling={"energy": 1, "force": 10},
    )
    trainer = pl.Trainer(max_steps=5, logger=False, enable_checkpointing=False)
    trainer.fit(task, datamodule=devset)
    # make sure losses are tracked
    for key in ["energy", "force"]:
        assert f"train_{key}" in trainer.logged_metrics

    # Scenario where one task_key is set. Expect to use one task_loss_scaling value.
    task = ForceRegressionTask(
        **egnn_config,
        task_keys=["force"],
        task_loss_scaling={"force": 10},
    )
    trainer = pl.Trainer(max_steps=5, logger=False, enable_checkpointing=False)
    trainer.fit(task, datamodule=devset)
    # make sure losses are tracked
    for key in ["force"]:
        assert f"train_{key}" in trainer.logged_metrics


def test_scalar_regression(egnn_config):
    devset = MatSciMLDataModule.from_devset(
        "NomadDataset",
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
    # Scenario where task_keys are set, and not all loss scaling values are set.
    # Expect to use fill in task_loss_scaling with default 1 for missing key.
    task = ScalarRegressionTask(
        **egnn_config,
        task_keys=["efermi", "energy_total", "relative_energy"],
        task_loss_scaling={"efermi": 1e-5, "energy_total": 1e4},
    )
    trainer = pl.Trainer(max_steps=5, logger=False, enable_checkpointing=False)
    trainer.fit(task, datamodule=devset)
    # make sure losses are tracked
    for key in ["efermi", "energy_total"]:
        assert f"train_{key}" in trainer.logged_metrics

    task = ScalarRegressionTask(
        **egnn_config,
        task_keys=["efermi", "energy_total", "relative_energy"],
    )
    trainer = pl.Trainer(max_steps=5, logger=False, enable_checkpointing=False)
    trainer.fit(task, datamodule=devset)
    # make sure losses are tracked
    for key in ["efermi", "energy_total"]:
        assert f"train_{key}" in trainer.logged_metrics
