import pytest

from ocpmodels.lightning.data_utils import MatSciMLDataModule
from ocpmodels.datasets import __all__


dset_names = list(
    sorted(filter(lambda x: "Dataset" in x and "Multi" not in x, __all__))
)


@pytest.mark.parametrize("dset_classname", dset_names)
def test_datamodule_devset(dset_classname: str):
    """
    This tests the end-to-end data module workflow with devsets.

    The end result should be a batch of data.
    """
    datamodule = MatSciMLDataModule.from_devset(dset_classname)
    datamodule.setup()
    assert next(iter(datamodule.train_dataloader()))
