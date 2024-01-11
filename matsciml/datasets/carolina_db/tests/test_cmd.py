from __future__ import annotations

import shutil

import pytest

from matsciml.datasets.carolina_db import CMDRequest
from matsciml.datasets.carolina_db.dataset import CMDataset

TEST_IDS = [0, 1, 2]


@pytest.fixture(scope="session")
def devset_dir(tmp_path_factory):
    devset_dir = tmp_path_factory.mktemp("test_lmdb")
    yield devset_dir
    shutil.rmtree(devset_dir)


@pytest.fixture
@pytest.mark.dependency(depends=["devset_dir"])
def cmd_module(devset_dir):
    cmd = CMDRequest()
    cmd.material_ids = TEST_IDS
    cmd.data_dir = devset_dir
    return cmd


@pytest.mark.dependency()
@pytest.mark.carolina_api
def test_download_data(cmd_module):
    request_status = cmd_module.cmd_request()
    assert all(request_status.values())


@pytest.mark.dependency(depends=["test_download_data"])
@pytest.mark.carolina_api
def test_process_data(cmd_module):
    data = cmd_module.process_data()
    assert len(data) != 0
    assert all(entry is not None for entry in data)
    datum = data.pop(0)
    assert datum.get("cart_coords") is not None
    assert datum.get("energy") is not None


@pytest.mark.dependency(depends=["test_process_data"])
@pytest.mark.mp_api
def test_serialize_lmdb(cmd_module):
    data = cmd_module.process_data()
    cmd_module.to_lmdb(cmd_module.data_dir)


@pytest.mark.dependency(depends=["test_serialize_lmdb"])
@pytest.mark.local
def test_dataset_load(devset_dir):
    dset = CMDataset(devset_dir)
    for index in range(3):
        data = dset.__getitem__(index)
        assert all(
            [
                key in data.keys()
                for key in ["pos", "atomic_numbers", "pc_features", "dataset"]
            ],
        )


@pytest.mark.dependency(depends=["test_dataset_load"])
@pytest.mark.local
def test_dataset_collate(devset_dir):
    dset = CMDataset(devset_dir)
    data = [dset.__getitem__(index) for index in range(len(TEST_IDS))]
    batch = dset.collate_fn(data)
    # check the nuclear coordinates and numbers match what is expected
    assert batch["pos"].size(0) == sum(batch["sizes"])
    assert batch["pos"].ndim == 2
    assert batch["atomic_numbers"].size(0) == len(TEST_IDS)
    assert batch["atomic_numbers"].ndim == 2


@pytest.mark.dependency(depends=["test_dataset_load"])
@pytest.mark.local
def test_dataset_target_keys(devset_dir):
    # this tests target key property without manually grabbing a batch
    dset = CMDataset(devset_dir)
    assert dset.target_keys == {"regression": ["energy"]}


def test_saved_devset():
    dset = CMDataset(str(CMDataset.__devset__))
    samples = [dset.__getitem__(i) for i in range(16)]
    batch = dset.collate_fn(samples)
    assert all([key in batch for key in ["pos", "pc_features", "mask", "targets"]])
