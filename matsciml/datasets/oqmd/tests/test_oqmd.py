from __future__ import annotations

import os
import shutil

import pytest

from matsciml.datasets.oqmd import OQMDRequest
from matsciml.datasets.oqmd.dataset import OQMDDataset

TEST_IDS = [0, 1, 2, 3, 4]


@pytest.fixture(scope="session")
def devset_dir(tmp_path_factory):
    devset_dir = tmp_path_factory.mktemp("test_lmdb")
    yield devset_dir
    shutil.rmtree(devset_dir)


@pytest.fixture
@pytest.mark.dependency(depends=["devset_dir"])
def oqmd_module(devset_dir):
    cmd = OQMDRequest(num_workers=1)
    cmd.material_ids = TEST_IDS
    cmd.limit = 1
    cmd.data_dir = devset_dir
    return cmd


@pytest.mark.dependency()
@pytest.mark.oqmd_api
def test_download_data(oqmd_module):
    request_status = oqmd_module.oqmd_request()
    assert all(request_status.values())


@pytest.mark.dependency(depends=["test_download_dadta"])
@pytest.mark.mp_api
def test_process_json(oqmd_module):
    request_status = oqmd_module.oqmd_request()
    oqmd_module.process_json()
    assert all([sample.get("cart_coords", False) for sample in oqmd_module.data])


@pytest.mark.dependency(depends=["test_download_dadta"])
@pytest.mark.mp_api
def test_serialize_lmdb(oqmd_module):
    request_status = oqmd_module.oqmd_request()
    oqmd_module.process_json()
    oqmd_module.to_lmdb(oqmd_module.data_dir)


@pytest.mark.dependency(depends=["test_serialize_lmdb"])
@pytest.mark.local
def test_dataset_load(devset_dir):
    dset = OQMDDataset(devset_dir)
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
    dset = OQMDDataset(devset_dir)
    data = [dset.__getitem__(index) for index in range(len(TEST_IDS))]
    batch = dset.collate_fn(data)
    # check the nuclear coordinates and numbers match what is expected
    assert batch["pos"].size(0) == sum(batch["sizes"])
    assert batch["pos"].ndim == 2
    assert len(batch["atomic_numbers"]) == len(TEST_IDS)


@pytest.mark.dependency(depends=["test_dataset_load"])
@pytest.mark.local
def test_dataset_target_keys(devset_dir):
    # this tests target key property without manually grabbing a batch
    dset = OQMDDataset(devset_dir)
    assert dset.target_keys == {
        "regression": ["energy", "band_gap", "stability"],
    }


def test_saved_devset():
    dset = OQMDDataset(str(OQMDDataset.__devset__))
    samples = [dset.__getitem__(i) for i in range(16)]
    batch = dset.collate_fn(samples)
    assert all([key in batch for key in ["pos", "pc_features", "mask", "targets"]])
