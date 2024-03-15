import pytest
import pytorch_lightning as pl

from matsciml.datasets.transforms import (
    PeriodicPropertiesTransform,
    PointCloudToGraphTransform,
    MGLDataTransform,
)
from matsciml.lightning import MatSciMLDataModule
from matsciml.common.registry import registry
from matsciml.models import TensorNet
from matsciml.datasets.utils import element_types
from matsciml.models.base import ForceRegressionTask, GradFreeForceRegressionTask

import torch


# fixture for some nominal set of hyperparameters that can be used
# across datasets
@pytest.fixture
def model_fixture() -> TensorNet:
    model = TensorNet(element_types=element_types())
    return model


@pytest.fixture
def devset_fixture() -> MatSciMLDataModule:
    devset = MatSciMLDataModule.from_devset(
        "S2EFDataset",
        dset_kwargs={
            "transforms": [
                PeriodicPropertiesTransform(cutoff_radius=10.0),
                PointCloudToGraphTransform(
                    "dgl",
                    cutoff_dist=20.0,
                    node_keys=["pos", "atomic_numbers"],
                ),
            ],
        },
    )
    return devset


# here we filter out datasets from the registry that don't make sense
ignore_dset = ["Multi", "PyG", "Cdvae"]
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
def test_model_forward_nograd(dset_class_name: str, model_fixture: TensorNet):
    transforms = [
        PeriodicPropertiesTransform(cutoff_radius=6.0),
        PointCloudToGraphTransform("dgl"),
        MGLDataTransform(),
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
        embeddings = model_fixture(batch)
    # returns embeddings, and runs numerical checks
    for z in [embeddings.system_embedding, embeddings.point_embedding]:
        assert torch.isreal(z).all()
        assert ~torch.isnan(z).all()  # check there are no NaNs
        assert torch.isfinite(z).all()
        assert torch.all(torch.abs(z) <= 1000)  # ensure reasonable values


def test_force_regression(model_fixture, devset_fixture):
    task = ForceRegressionTask(
        model_fixture, output_kwargs={"lazy": False, "input_dim": 64, "hidden_dim": 64}
    )
    trainer = pl.Trainer(fast_dev_run=10)
    trainer.fit(task, datamodule=devset_fixture)
    # make sure losses are tracked
    for key in ["energy", "force"]:
        assert f"train_{key}" in trainer.logged_metrics


def test_gradfree_force_regression(model_fixture, devset_fixture):
    task = GradFreeForceRegressionTask(
        model_fixture,
        output_kwargs={"lazy": False, "input_dim": 64, "hidden_dim": 64},
    )
    trainer = pl.Trainer(fast_dev_run=10)
    trainer.fit(task, datamodule=devset_fixture)
    # make sure losses are tracked
    assert "train_force" in trainer.logged_metrics
