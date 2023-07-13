import pytest

from ocpmodels.lightning.data_utils import MatSciMLDataModule
from ocpmodels.common.registry import registry
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
    # test the data loaders
    assert next(iter(datamodule.train_dataloader()))
    assert next(iter(datamodule.val_dataloader()))
    assert next(iter(datamodule.test_dataloader()))


@pytest.mark.parametrize("dset_classname", dset_names)
def test_datamodule_manual_trainonly(dset_classname):
    dset = registry.get_dataset_class(dset_classname)
    datamodule = MatSciMLDataModule(
        dataset=dset_classname,
        train_path=dset.__devset__,
        batch_size=8,
    )
    datamodule.setup()
    assert next(iter(datamodule.train_dataloader()))


@pytest.mark.parametrize("dset_classname", dset_names)
def test_datamodule_manual_splits(dset_classname):
    dset = registry.get_dataset_class(dset_classname)
    datamodule = MatSciMLDataModule(
        dataset=dset_classname, train_path=dset.__devset__, batch_size=8, val_split=0.2
    )
    datamodule.setup()
    assert next(iter(datamodule.train_dataloader()))
    assert next(iter(datamodule.val_dataloader()))


def test_bad_dataset():
    # this makes sure that any willy nilly dataset will fail
    datamodule = MatSciMLDataModule(
        dataset="BadDataset", train_path="/not/a/path", batch_size=8
    )
    with pytest.raises(KeyError):
        datamodule.setup()
