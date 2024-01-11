from __future__ import annotations

from itertools import product

import pytest

from matsciml.common import package_registry
from matsciml.common.registry import registry
from matsciml.datasets import __all__
from matsciml.datasets.transforms import (
    GraphToPointCloudTransform,
    PointCloudToGraphTransform,
)
from matsciml.lightning.data_utils import MatSciMLDataModule

dset_names = list(
    sorted(
        filter(
            lambda x: "Dataset" in x and "Multi" not in x and "Synthetic" not in x,
            __all__,
        ),
    ),
)

not_ocp = list(filter(lambda x: "IS2RE" not in x and "S2EF" not in x, dset_names))
just_ocp = list(filter(lambda x: "IS2RE" in x or "S2EF" in x, dset_names))


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
        dataset=dset_classname,
        train_path=dset.__devset__,
        batch_size=8,
        val_split=0.2,
    )
    datamodule.setup()
    assert next(iter(datamodule.train_dataloader()))
    assert next(iter(datamodule.val_dataloader()))


@pytest.mark.parametrize(
    "dset_classname, backend",
    list(product(not_ocp, ["dgl", "pyg"])),
)
def test_datamodule_graph_transforms(dset_classname, backend):
    if package_registry[backend]:
        t = PointCloudToGraphTransform(backend)
        datamodule = MatSciMLDataModule.from_devset(
            dset_classname,
            dset_kwargs={"transforms": [t]},
        )
        datamodule.setup()
        check_keys = ["pos", "atomic_numbers"]
        for split in ["train", "val", "test"]:
            loader = getattr(datamodule, f"{split}_dataloader")()
            batch = next(iter(loader))
            assert "graph" in batch
            if backend == "dgl":
                target = batch["graph"].ndata
            else:
                target = batch["graph"]
            assert all([key in target for key in check_keys])


@pytest.mark.parametrize("dset_classname", just_ocp)
def test_ocp_pc_transforms(dset_classname):
    t = GraphToPointCloudTransform("dgl")
    datamodule = MatSciMLDataModule.from_devset(
        dset_classname,
        dset_kwargs={"transforms": [t]},
    )
    datamodule.setup()
    for split in ["train", "val", "test"]:
        loader = getattr(datamodule, f"{split}_dataloader")()
        batch = next(iter(loader))
        assert "graph" not in batch
        assert "pc_features" in batch


def test_bad_dataset():
    # this makes sure that any willy nilly dataset will fail
    datamodule = MatSciMLDataModule(
        dataset="BadDataset",
        train_path="/not/a/path",
        batch_size=8,
    )
    with pytest.raises(KeyError):
        datamodule.setup()
