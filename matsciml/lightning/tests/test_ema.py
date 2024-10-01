from __future__ import annotations

import pytorch_lightning as pl
import pytest

from matsciml.datasets.transforms import (
    PeriodicPropertiesTransform,
    PointCloudToGraphTransform,
    FrameAveraging,
    UnitCellCalculator,
)

from matsciml.lightning import MatSciMLDataModule, MultiDataModule
from matsciml.datasets import MultiDataset, IS2REDataset, S2EFDataset
from matsciml.models.pyg import EGNN
from matsciml.lightning.callbacks import ExponentialMovingAverageCallback
from matsciml.models.pyg import FAENet
from torch import nn
from e3nn.o3 import Irreps
from mace.modules.blocks import RealAgnosticInteractionBlock
from matsciml.models.pyg.mace import MACEWrapper
from matsciml.models.dgl import PLEGNNBackbone
from matsciml.models.base import (
    MultiTaskLitModule,
    ForceRegressionTask,
    ScalarRegressionTask,
)


def test_egnn_end_to_end_with_ema():
    """
    Test the end to end pipeline using a devset with EGNN and ema callback.

    The idea is that this basically mimics an example script to
    try and maximize coverage across dataset to training, which
    is particularly useful for checking new dependencies, etc.
    """
    dm = MatSciMLDataModule.from_devset(
        "MaterialsProjectDataset",
        dset_kwargs={
            "transforms": [
                PeriodicPropertiesTransform(5.0, adaptive_cutoff=True),
                PointCloudToGraphTransform("pyg"),
            ]
        },
        batch_size=8,
    )

    # this specifies a whole lot to make sure we have coverage
    task = ScalarRegressionTask(
        encoder_class=EGNN,
        encoder_kwargs={
            "hidden_dim": 48,
            "output_dim": 32,
            "num_conv": 2,
            "num_atom_embedding": 200,
        },
        scheduler_kwargs={
            "CosineAnnealingLR": {
                "T_max": 5,
                "eta_min": 1e-7,
            }
        },
        lr=1e-3,
        weight_decay=0.0,
        output_kwargs={
            "lazy": False,
            "hidden_dim": 48,
            "input_dim": 48,
            "dropout": 0.2,
            "num_hidden": 2,
        },
        task_keys=["band_gap"],
    )

    trainer = pl.Trainer(fast_dev_run=5, callbacks=[ExponentialMovingAverageCallback()])
    trainer.fit(task, datamodule=dm)
    assert hasattr(task, "ema_module")


def test_lazy_fail():
    """Lazy modules are not supported right now, so this ensures they fail"""
    dm = MatSciMLDataModule.from_devset(
        "MaterialsProjectDataset",
        dset_kwargs={
            "transforms": [
                PeriodicPropertiesTransform(5.0, adaptive_cutoff=True),
                PointCloudToGraphTransform("pyg"),
            ]
        },
        batch_size=8,
    )

    # this specifies a whole lot to make sure we have coverage
    task = ScalarRegressionTask(
        encoder_class=EGNN,
        encoder_kwargs={
            "hidden_dim": 48,
            "output_dim": 32,
            "num_conv": 2,
            "num_atom_embedding": 200,
        },
        scheduler_kwargs={
            "CosineAnnealingLR": {
                "T_max": 5,
                "eta_min": 1e-7,
            }
        },
        lr=1e-3,
        weight_decay=0.0,
        output_kwargs={"lazy": True},
        task_keys=["band_gap"],
    )

    trainer = pl.Trainer(fast_dev_run=5, callbacks=[ExponentialMovingAverageCallback()])
    with pytest.raises(RuntimeError):
        trainer.fit(task, datamodule=dm)


def test_mace_with_ema():
    """
    Test the MACE Wrapper with ema callback.
    """
    # Construct MACE relaxed energy regression with PyG implementation of E(n)-GNN
    task = ScalarRegressionTask(
        encoder_class=MACEWrapper,
        encoder_kwargs={
            "r_max": 6.0,
            "num_bessel": 3,
            "num_polynomial_cutoff": 3,
            "max_ell": 2,
            "interaction_cls": RealAgnosticInteractionBlock,
            "interaction_cls_first": RealAgnosticInteractionBlock,
            "num_interactions": 2,
            "atom_embedding_dim": 64,
            "MLP_irreps": Irreps("256x0e"),
            "avg_num_neighbors": 10.0,
            "correlation": 1,
            "radial_type": "bessel",
            "gate": nn.Identity(),
            "distance_transform": None,
        },
        task_keys=["energy_relaxed"],
        output_kwargs={"lazy": False, "hidden_dim": 128, "input_dim": 128},
    )

    # Prepare data module
    dm = MatSciMLDataModule.from_devset(
        "IS2REDataset",
        dset_kwargs={
            "transforms": [
                PeriodicPropertiesTransform(5.0, adaptive_cutoff=True),
                PointCloudToGraphTransform(
                    "pyg",
                    node_keys=["pos", "atomic_numbers"],
                ),
            ],
        },
    )

    # Run a quick training loop
    trainer = pl.Trainer(
        fast_dev_run=5, callbacks=[ExponentialMovingAverageCallback(0.999)]
    )
    trainer.fit(task, datamodule=dm)
    assert hasattr(task, "ema_module")


def test_faenet_with_ema():
    """
    Test FAENet with ema Callback.
    """
    task = ScalarRegressionTask(
        encoder_class=FAENet,
        encoder_kwargs={
            "pred_as_dict": False,
            "hidden_dim": 128,
            "out_dim": 64,
            "tag_hidden_channels": 0,
            "input_dim": 128,
        },
        output_kwargs={"lazy": False, "input_dim": 64, "hidden_dim": 64},
        task_keys=["band_gap"],
    )

    dm = MatSciMLDataModule.from_devset(
        "MaterialsProjectDataset",
        dset_kwargs={
            "transforms": [
                UnitCellCalculator(),
                PointCloudToGraphTransform(
                    "pyg",
                    cutoff_dist=20.0,
                    node_keys=["pos", "atomic_numbers"],
                ),
                FrameAveraging(frame_averaging="3D", fa_method="stochastic"),
            ],
        },
    )

    # run a quick training loop
    trainer = pl.Trainer(fast_dev_run=5, callbacks=[ExponentialMovingAverageCallback()])
    trainer.fit(task, datamodule=dm)
    assert hasattr(task, "ema_module")


def test_force_regression_with_ema():
    """
    Tests force regression with ema using PLEGNNBackbone.
    """
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

    task = ForceRegressionTask(
        encoder_class=PLEGNNBackbone,
        encoder_kwargs=model_args,
        output_kwargs={"lazy": False, "hidden_dim": 128, "input_dim": 128},
    )
    trainer = pl.Trainer(fast_dev_run=5, callbacks=[ExponentialMovingAverageCallback()])
    trainer.fit(task, datamodule=devset)
    # make sure losses are tracked
    for key in ["energy", "force"]:
        assert f"train_{key}" in trainer.logged_metrics
    assert hasattr(task, "ema_module")


def test_multitask_ema():
    transforms = [
        PeriodicPropertiesTransform(6.0, adaptive_cutoff=True),
        PointCloudToGraphTransform(
            "dgl",
            cutoff_dist=6.0,
            node_keys=["pos", "atomic_numbers"],
        ),
    ]
    dm = MultiDataModule(
        train_dataset=MultiDataset(
            [
                IS2REDataset.from_devset(transforms=transforms),
                S2EFDataset.from_devset(transforms=transforms),
            ],
        ),
        batch_size=8,
    )
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

    is2re = ScalarRegressionTask(
        encoder_class=PLEGNNBackbone,
        encoder_kwargs=model_args,
        task_keys=["energy_init", "energy_relaxed"],
        output_kwargs={"lazy": False, "hidden_dim": 128, "input_dim": 128},
    )
    s2ef = ForceRegressionTask(
        encoder_class=PLEGNNBackbone,
        encoder_kwargs=model_args,
        task_keys=["energy", "force"],
        output_kwargs={"lazy": False, "hidden_dim": 128, "input_dim": 128},
    )

    task = MultiTaskLitModule(
        ("IS2REDataset", is2re),
        ("S2EFDataset", s2ef),
    )
    trainer = pl.Trainer(
        fast_dev_run=5, callbacks=ExponentialMovingAverageCallback(0.9)
    )
    trainer.fit(task, datamodule=dm)
    assert hasattr(task, "ema_module")
