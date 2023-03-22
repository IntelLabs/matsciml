import pytest

from ocpmodels.datasets.materials_project import (
    MaterialsProjectRequest,
    MaterialsProjectDataset,
)

# TODO add marks to pyproject.toml


@pytest.fixture
def dev_request():
    return MaterialsProjectRequest.devset()


@pytest.mark.dependency()
@pytest.mark.mp_api
def test_devset(dev_request):
    data = dev_request.retrieve_data()
    assert len(dev_request.data) != 0
    # pop an entry and make sure it has keys
    datum = data.pop(0)
    assert getattr(datum, "band_gap") is not None
    assert hasattr(datum, "structure")


@pytest.mark.dependency(depends=["test_devset"])
@pytest.mark.mp_api
def test_serialize_lmdb(dev_request):
    data = dev_request.retrieve_data()
    dev_request.to_lmdb("test_lmdb")


@pytest.mark.dependency(depends=["test_serialize_lmdb"])
@pytest.mark.local
def test_dataset_load():
    dset = MaterialsProjectDataset("test_lmdb")
    for index in range(10):
        data = dset.__getitem__(index)
        assert all([key in data.keys() for key in ["pos", "atomic_numbers", "lattice_features"]])
