
import pytest
import os

from ocpmodels.lightning.data_utils import MaterialsProjectDataModule

@pytest.mark.dependency()
def test_pc_setup():
    dset = MaterialsProjectDataModule.from_devset(graphs=False)
    # set random seed to test
    os.environ["PL_GLOBAL_SEED"] = "10241"
    dset.setup()
    train_loader = dset.train_dataloader()
    batch = next(iter(train_loader))


@pytest.mark.dependency(depends=["test_pc_setup"])
def test_pc_val_setup():
    dset = MaterialsProjectDataModule.from_devset(graphs=False, val_split=0.2)
    dset.setup()
    assert len(dset.splits["val"]) > 0, f"Expecting non-zero validation {len(dset.splits['val'])}"
    # get loader now
    val_loader = dset.val_dataloader()
    batch = next(iter(val_loader))


@pytest.mark.dependency(depends=["test_pc_setup"])
def test_pc_test_setup():
    dset = MaterialsProjectDataModule.from_devset(graphs=False, test_split=0.2)
    dset.setup()
    assert len(dset.splits["test"]) > 0, f"Expecting non-zero validation {len(dset.splits['test'])}"
    # get loader now
    test_loader = dset.test_dataloader()
    batch = next(iter(test_loader))


@pytest.mark.dependency(depends=["test_pc_val_setup", "test_pc_test_setup"])
def test_pc_all_setup():
    dset = MaterialsProjectDataModule.from_devset(graphs=False, test_split=0.2, val_split=0.2)
    dset.setup()
    for key in ["train", "val", "test", "predict"]:
        loader = getattr(dset, f"{key}_dataloader")()
        batch = next(iter(loader))


@pytest.mark.dependency(depends=["test_pc_all_setup"])
def test_pc_no_args():
    # this makes sure that when nothing is provided, the code breaks
    try:
        dm = MaterialsProjectDataModule()
    except AssertionError:
        assert True


@pytest.mark.dependency()
def test_graph_setup():
    dset = MaterialsProjectDataModule.from_devset(graphs=True)
    # set random seed to test
    os.environ["PL_GLOBAL_SEED"] = "10241"
    dset.setup()
    train_loader = dset.train_dataloader()
    batch = next(iter(train_loader))


@pytest.mark.dependency
def test_graph_all_setup():
    dset = MaterialsProjectDataModule.from_devset(graphs=True, test_split=0.2, val_split=0.2)
    dset.setup()
    for key in ["train", "val", "test", "predict"]:
        loader = getattr(dset, f"{key}_dataloader")()
        batch = next(iter(loader))


@pytest.mark.dependency(depends=["test_pc_setup"])
def test_mp_target_keys():
    dset = MaterialsProjectDataModule.from_devset()
    assert "regression" in dset.target_keys
    assert "band_gap" in dset.target_keys["regression"]
