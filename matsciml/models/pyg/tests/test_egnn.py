from __future__ import annotations

import pytest
import torch

# this import is not used, but ensures that the registry is updated
from matsciml import datasets
from matsciml.common.registry import registry
from matsciml.datasets.transforms import (
    PeriodicPropertiesTransform,
    PointCloudToGraphTransform,
)
from matsciml.lightning import MatSciMLDataModule
from matsciml.models.pyg import EGNN


@pytest.fixture
def egnn_architecture() -> EGNN:
    """
    Fixture for a nominal EGNN architecture.

    Some lightweight (but realistic) hyperparameters are
    used to test data flowing through the model.

    Returns
    -------
    EGNN
        Concrete EGNN object
    """
    model = EGNN(64, 16)
    return model


# here we filter out datasets from the registry that don't make sense
ignore_dset = ["Multi", "M3G", "PyG", "Cdvae"]
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
def test_model_forward_nograd(dset_class_name: str, egnn_architecture: EGNN):
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
    egnn_architecture : EGNN
        Concrete EGNN object with some parameters
    """
    transforms = [
        PeriodicPropertiesTransform(cutoff_radius=6.0),
        PointCloudToGraphTransform("pyg"),
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
        embeddings = egnn_architecture(batch)
    # returns embeddings, and runs numerical checks
    for z in [embeddings.system_embedding, embeddings.point_embedding]:
        assert torch.isreal(z).all()
        assert ~torch.isnan(z).all()  # check there are no NaNs
        assert torch.isfinite(z).all()
        assert torch.all(torch.abs(z) <= 1000)  # ensure reasonable values
