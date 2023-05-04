
import pytest
import os

from ocpmodels.lightning.data_utils import LiPSDataModule

@pytest.mark.dependency()
def test_pc_setup():
    dset = LiPSDataModule.from_devset(graphs=False)
    # set random seed to test
    os.environ["PL_GLOBAL_SEED"] = "10241"
    dset.setup()
    train_loader = dset.train_dataloader()
    batch = next(iter(train_loader))


@pytest.mark.dependency(depends=["test_pc_setup"])
def test_pc_val_setup():
    dset = LiPSDataModule.from_devset(graphs=False, val_split=0.2)
    dset.setup()
    assert len(dset.splits["val"]) > 0, f"Expecting non-zero validation {len(dset.splits['val'])}"
    # get loader now
    val_loader = dset.val_dataloader()
    batch = next(iter(val_loader))


@pytest.mark.dependency(depends=["test_pc_setup"])
def test_pc_test_setup():
    dset = LiPSDataModule.from_devset(graphs=False, test_split=0.2)
    dset.setup()
    assert len(dset.splits["test"]) > 0, f"Expecting non-zero validation {len(dset.splits['test'])}"
    # get loader now
    test_loader = dset.test_dataloader()
    batch = next(iter(test_loader))


@pytest.mark.dependency(depends=["test_pc_val_setup", "test_pc_test_setup"])
def test_pc_all_setup():
    dset = LiPSDataModule.from_devset(graphs=False, test_split=0.2, val_split=0.2)
    dset.setup()
    for key in ["train", "val", "test", "predict"]:
        loader = getattr(dset, f"{key}_dataloader")()
        batch = next(iter(loader))


@pytest.mark.dependency()
def test_graph_setup():
    dset = LiPSDataModule.from_devset(graphs=True)
    # set random seed to test
    os.environ["PL_GLOBAL_SEED"] = "10241"
    dset.setup()
    train_loader = dset.train_dataloader()
    batch = next(iter(train_loader))


@pytest.mark.dependency
def test_graph_all_setup():
    dset = LiPSDataModule.from_devset(graphs=True, test_split=0.2, val_split=0.2)
    dset.setup()
    for key in ["train", "val", "test", "predict"]:
        loader = getattr(dset, f"{key}_dataloader")()
        batch = next(iter(loader))


@pytest.mark.dependency(depends=["test_pc_setup"])
def test_datamodule_target_keys():
    dset = LiPSDataModule.from_devset()
    assert dset.target_keys == ["energy", "force"]

