import shutil

import pytest

from ocpmodels.datasets.nomad import NomadRequest
from ocpmodels.datasets.nomad.dataset import NomadDataset

TEST_IDS = {
    0: "omTIFQFoC_ryxWm61HvGfG31Y_xq",
    1: "vO1djw22GPm9CJcckNyPy1JsS9mb",
    2: "r31Xq3nPTsEl35wLoAfqH0eXp_Ve",
    3: "kc-0nyFuX3zmx8FaHRrMCLgfTEcr",
    4: "iH0lS5fum5uG_ZxKFqVFrpM1t-Vn",
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


# @pytest.mark.dependency(depends=["test_serialize_lmdb"])
# @pytest.mark.local
# def test_dataset_load(devset_dir):
#     dset = NomadDataset(devset_dir)
#     for index in range(3):
#         data = dset.__getitem__(index)
#         import pdb; pdb.set_trace()
#         assert all(
#             [
#                 key in data.keys()
#                 for key in ["pos", "atomic_numbers", "pc_features", "dataset"]
#             ]
#         )


# @pytest.mark.dependency(depends=["test_dataset_load"])
# @pytest.mark.local
# def test_dataset_collate(devset_dir):
#     dset = NomadDataset(devset_dir)
#     data = [dset.__getitem__(index) for index in range(len(TEST_IDS))]
#     batch = dset.collate_fn(data)
#     # check the nuclear coordinates and numbers match what is expected
#     assert batch["pos"].size(0) == sum(batch["sizes"])
#     assert batch["pos"].ndim == 2
#     assert batch["atomic_numbers"].size(0) == len(TEST_IDS)
#     assert batch["atomic_numbers"].ndim == 2


# @pytest.mark.dependency(depends=["test_dataset_load"])
# @pytest.mark.local
# def test_dataset_target_keys(devset_dir):
#     # this tests target key property without manually grabbing a batch
#     dset = NomadDataset(devset_dir)
#     assert dset.target_keys == {"regression": ["energy"]}


# def test_saved_devset():
#     dset = NomadDataset(str(NomadDataset.__devset__))
#     samples = [dset.__getitem__(i) for i in range(16)]
#     batch = dset.collate_fn(samples)
#     assert all([key in batch for key in ["pos", "pc_features", "mask", "targets"]])
