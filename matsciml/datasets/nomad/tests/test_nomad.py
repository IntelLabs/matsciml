from __future__ import annotations

import shutil

import pytest

from matsciml.datasets.nomad import NomadRequest
from matsciml.datasets.nomad.dataset import NomadDataset

TEST_IDS = {
    0: "GjAKByPxraKfkFCdFrwp0omDVQZ7",
    1: "0FwC9lqZWvGigWMtxgdn7M6YXhwu",
    2: "VSRNiGFB2epCnn6OBY04S4175SIY",
    3: "wvfvLz6S0xj7S8oXVIpEbDdh1hwD",
    4: "OldNS7xP3AtG_NT3uFEyrlk1xh20",
}


@pytest.fixture(scope="session")
def devset_dir(tmp_path_factory):
    devset_dir = tmp_path_factory.mktemp("test_lmdb")
    yield devset_dir
    shutil.rmtree(devset_dir)


@pytest.fixture
@pytest.mark.dependency(depends=["devset_dir"])
def nomad_module(devset_dir):
    cmd = NomadRequest(num_workers=1)
    cmd.material_ids = TEST_IDS
    cmd.data_dir = devset_dir
    return cmd


@pytest.mark.dependency()
@pytest.mark.carolina_api
def test_download_data(nomad_module):
    data = nomad_module.nomad_request()
    assert None not in data
    datum = data.pop(0)
    assert datum.get("properties") is not None
    assert datum.get("material") is not None


@pytest.mark.dependency(depends=["test_download_dadta"])
@pytest.mark.mp_api
def test_serialize_lmdb(nomad_module):
    data = nomad_module.nomad_request()
    nomad_module.to_lmdb(nomad_module.data_dir)


@pytest.mark.dependency(depends=["test_serialize_lmdb"])
@pytest.mark.local
def test_dataset_load(devset_dir):
    dset = NomadDataset(devset_dir)
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
    dset = NomadDataset(devset_dir)
    data = [dset.__getitem__(index) for index in range(len(TEST_IDS))]
    batch = dset.collate_fn(data)
    # check the nuclear coordinates and numbers match what is expected
    assert batch["pos"].size(0) == sum(batch["sizes"])
    assert batch["pos"].ndim == 2
    assert len(batch["atomic_numbers"]) == len(TEST_IDS)
    # assert batch["atomic_numbers"].ndim == 2


@pytest.mark.dependency(depends=["test_dataset_load"])
@pytest.mark.local
def test_dataset_target_keys(devset_dir):
    # this tests target key property without manually grabbing a batch
    dset = NomadDataset(devset_dir)
    assert dset.target_keys == {
        "regression": ["energy_total", "efermi"],
        "classification": ["spin_polarized"],
    }


def test_saved_devset():
    dset = NomadDataset(str(NomadDataset.__devset__))
    samples = [dset.__getitem__(i) for i in range(16)]
    batch = dset.collate_fn(samples)
    assert all([key in batch for key in ["pos", "pc_features", "mask", "targets"]])
