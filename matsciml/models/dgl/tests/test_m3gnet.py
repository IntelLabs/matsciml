import pytest

from matsciml.datasets.transforms import (
    PeriodicPropertiesTransform,
    PointCloudToGraphTransform,
    MGLDataTransform,
)
from matsciml.lightning import MatSciMLDataModule
from matsciml.common.registry import registry
from matsciml.models import M3GNet
from matsciml.datasets.utils import element_types

import torch


# fixture for some nominal set of hyperparameters that can be used
# across datasets
@pytest.fixture
def model_fixture() -> M3GNet:
    model = M3GNet(element_types=element_types(), return_all_layer_output=True)
    return model


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
def test_model_forward_nograd(dset_class_name: str, model_fixture: M3GNet):
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
