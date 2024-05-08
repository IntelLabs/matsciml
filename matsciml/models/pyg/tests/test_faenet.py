from __future__ import annotations

import pytest
import torch
import pytorch_lightning as pl

# this import is not used, but ensures that the registry is updated
from matsciml.common.registry import registry
from matsciml.datasets.transforms import (
    PeriodicPropertiesTransform,
    PointCloudToGraphTransform,
    FrameAveraging,
)
from matsciml.lightning import MatSciMLDataModule
from matsciml.models.pyg import FAENet
from matsciml.models.base import ForceRegressionTask


@pytest.fixture
def faenet_architecture() -> FAENet:
    """
    Fixture for a nominal FAENet architecture.

    Some lightweight (but realistic) hyperparameters are
    used to test data flowing through the model.

    Returns
    -------
    FAENet
        Concrete FAENet object
    """
    faenet_kwargs = {
        "average_frame_embeddings": False,  # set to false for use with FA transform
        "pred_as_dict": False,
        "hidden_dim": 128,
        "out_dim": 128,
        "tag_hidden_channels": 0,
    }
    model = FAENet(**faenet_kwargs)
    return model


# here we filter out datasets from the registry that don't make sense
ignore_dset = ["Multi", "M3G", "PyG", "Cdvae", "SyntheticPointGroupDataset"]
filtered_list = list(
    filter(
        lambda x: all([target_str not in x for target_str in ignore_dset]),
        registry.__entries__["datasets"].keys(),
    ),
)


@pytest.mark.parametrize(
    "dset_class_name",
    filtered_list,
)
def test_model_forward_nograd(dset_class_name: str, faenet_architecture: FAENet):
    # these are necessary for the model to work as intended
    """
    This test checks model ``forward`` compatibility with datasets.

    The test is parameterized to run on all datasets in the registry
    that have *not* been filtered out; this list should be sparse,
    as the idea is to maximize coverage and we can just ignore failing
    combinations if they do not make sense and we can at least be
    aware of them.

    Parameters
    ----------
    dset_class_name : str
        Name of the dataset class to retrieve
    faenet_architecture : FAENet
        Concrete FAENet object with some parameters
    """
    transforms = [
        PeriodicPropertiesTransform(cutoff_radius=6.0),
        PointCloudToGraphTransform(
            "pyg",
            node_keys=["pos", "atomic_numbers"],
        ),
        FrameAveraging(frame_averaging="3D", fa_method="stochastic"),
    ]
    dm = MatSciMLDataModule.from_devset(
        dset_class_name,
        batch_size=8,
        dset_kwargs={"transforms": transforms},
    )
    # dummy initialization
    dm.setup("fit")
    loader = dm.train_dataloader()
    batch = next(iter(loader))
    # run the model without gradient tracking
    with torch.no_grad():
        embeddings = faenet_architecture(batch)
    # returns embeddings, and runs numerical checks
    for z in [embeddings.system_embedding, embeddings.point_embedding]:
        assert torch.isreal(z).all()
        assert ~torch.isnan(z).all()  # check there are no NaNs
        assert torch.isfinite(z).all()
        assert torch.all(torch.abs(z) <= 1000)  # ensure reasonable values


def test_force_regression(faenet_architecture):
    devset = MatSciMLDataModule.from_devset(
        "S2EFDataset",
        dset_kwargs={
            "transforms": [
                PeriodicPropertiesTransform(cutoff_radius=6.0, adaptive_cutoff=True),
                PointCloudToGraphTransform(
                    "pyg",
                    node_keys=["pos", "atomic_numbers"],
                ),
                FrameAveraging(frame_averaging="3D", fa_method="stochastic"),
            ],
        },
    )
    task = ForceRegressionTask(
        faenet_architecture,
    )
    trainer = pl.Trainer(
        max_steps=5, logger=False, enable_checkpointing=False, accelerator="cpu"
    )
    trainer.fit(task, datamodule=devset)
    # make sure losses are tracked
    for key in ["energy", "force"]:
        assert f"train_{key}" in trainer.logged_metrics

    loader = devset.train_dataloader()
    batch = next(iter(loader))
    outputs = task(batch)
    assert outputs["energy"].size(0) == batch["natoms"].size(0)
    assert outputs["force"].size(0) == sum(batch["natoms"]).item()
