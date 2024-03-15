from __future__ import annotations

import pytest
import torch
from torch import nn
from e3nn.o3 import Irreps
from mace.modules.blocks import RealAgnosticInteractionBlock

# this import is not used, but ensures that the registry is updated
from matsciml import datasets  # noqa: F401
from matsciml.common.registry import registry
from matsciml.datasets.transforms import (
    PeriodicPropertiesTransform,
    PointCloudToGraphTransform,
)
from matsciml.lightning import MatSciMLDataModule
from matsciml.models.pyg.mace import MACEWrapper


@pytest.fixture
def mace_architecture() -> MACEWrapper:
    """
    Fixture for a nominal mace architecture.

    Some lightweight (but realistic) hyperparameters are
    used to test data flowing through the model.

    Returns
    -------
    mace
        Concrete mace object
    """
    model_config = {
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
    }
    model = MACEWrapper(**model_config)
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
def test_model_forward_nograd(dset_class_name: str, mace_architecture: MACEWrapper):
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
    mace_architecture : EGNN
        Concrete mace object with some parameters
    """
    transforms = [
        PeriodicPropertiesTransform(cutoff_radius=6.0, adaptive_cutoff=True),
        PointCloudToGraphTransform("pyg"),
    ]
    dm = MatSciMLDataModule.from_devset(
        dset_class_name,
        batch_size=4,
        dset_kwargs={"transforms": transforms},
    )
    # dummy initialization
    dm.setup("fit")
    loader = dm.train_dataloader()
    batch = next(iter(loader))
    # run the model without gradient tracking
    with torch.no_grad():
        embeddings = mace_architecture(batch)
    # returns embeddings, and runs numerical checks
    for z in [embeddings.system_embedding, embeddings.point_embedding]:
        assert torch.isreal(z).all()
        assert ~torch.isnan(z).all()  # check there are no NaNs
        assert torch.isfinite(z).all()
        assert torch.all(torch.abs(z) <= 1000)  # ensure reasonable values
