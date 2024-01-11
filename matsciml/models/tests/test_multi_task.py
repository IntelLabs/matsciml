from __future__ import annotations

import pytest
import pytorch_lightning as pl

from matsciml.datasets import IS2REDataset, S2EFDataset, is2re_devset, s2ef_devset
from matsciml.datasets.multi_dataset import MultiDataset
from matsciml.lightning.data_utils import MultiDataModule
from matsciml.models import PLEGNNBackbone
from matsciml.models.base import (
    ForceRegressionTask,
    MultiTaskLitModule,
    ScalarRegressionTask,
)


@pytest.fixture
def model_def():
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

    model = PLEGNNBackbone(**model_args)
    return model


@pytest.fixture
def is2re_s2ef() -> MultiDataModule:
    dm = MultiDataModule(
        train_dataset=MultiDataset(
            [IS2REDataset(is2re_devset), S2EFDataset(s2ef_devset)],
        ),
        batch_size=16,
    )
    return dm


@pytest.mark.dependency()
def test_target_keys(is2re_s2ef):
    dm = is2re_s2ef
    keys = dm.target_keys
    assert keys == {
        "IS2REDataset": {"regression": ["energy_init", "energy_relaxed"]},
        "S2EFDataset": {"regression": ["energy", "force"]},
    }


@pytest.mark.dependency(depends=["test_target_keys", "model_def"])
def test_multitask_init(is2re_s2ef, model_def):
    dm = is2re_s2ef
    encoder = model_def
    is2re = ScalarRegressionTask(encoder, task_keys=["energy_init", "energy_relaxed"])
    s2ef = ForceRegressionTask(encoder, task_keys=["energy"])

    # pass task keys to make sure output heads are created
    task = MultiTaskLitModule(
        ("IS2REDataset", is2re),
        ("S2EFDataset", s2ef),
        task_keys=dm.target_keys,
    )
    # make sure we have output heads
    assert hasattr(is2re, "output_heads")
    assert hasattr(s2ef, "output_heads")
    assert "energy_init" in is2re.output_heads
    assert "energy" in s2ef.output_heads


@pytest.mark.dependency(depends=["test_multitask_init"])
def test_multitask_static_end2end(is2re_s2ef, model_def):
    dm = is2re_s2ef

    encoder = model_def
    is2re = ScalarRegressionTask(encoder, task_keys=["energy_init", "energy_relaxed"])
    s2ef = ForceRegressionTask(encoder, task_keys=["energy", "force"])

    task = MultiTaskLitModule(("IS2REDataset", is2re), ("S2EFDataset", s2ef))
    trainer = pl.Trainer(logger=False, enable_checkpointing=False, fast_dev_run=1)
    trainer.fit(task, datamodule=dm)


@pytest.mark.skip(reason="Broken test.")
def test_multitask_dynamic_end2end(is2re_s2ef, model_def):
    dm = is2re_s2ef

    encoder = model_def
    is2re = ScalarRegressionTask(encoder)
    s2ef = ForceRegressionTask(encoder)

    task = MultiTaskLitModule(("IS2REDataset", is2re), ("S2EFDataset", s2ef))
    trainer = pl.Trainer(logger=False, enable_checkpointing=False, fast_dev_run=1)
    trainer.fit(task, datamodule=dm)
