from __future__ import annotations

import shutil

import pytest

from matsciml.datasets import transforms
from matsciml.datasets.materials_project import (
    MaterialsProjectDataset,
    MaterialsProjectRequest,
)

# TODO add marks to pyproject.toml


@pytest.fixture
def dev_request():
    return MaterialsProjectRequest.devset()


@pytest.fixture(scope="session")
def devset_dir(tmp_path_factory):
    devset_dir = tmp_path_factory.mktemp("test_lmdb")
    yield devset_dir
    shutil.rmtree(devset_dir)


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
def test_serialize_lmdb(dev_request, devset_dir):
    data = dev_request.retrieve_data()
    dev_request.to_lmdb(devset_dir)


@pytest.mark.dependency(depends=["test_serialize_lmdb"])
@pytest.mark.local
def test_dataset_load(devset_dir):
    dset = MaterialsProjectDataset(devset_dir)
    for index in range(10):
        data = dset.__getitem__(index)
        assert all(
            [
                key in data.keys()
                for key in ["pos", "atomic_numbers", "lattice_features", "dataset"]
            ],
        )


@pytest.mark.dependency(depends=["test_dataset_load"])
@pytest.mark.local
def test_dataset_collate(devset_dir):
    dset = MaterialsProjectDataset(devset_dir)
    data = [dset.__getitem__(index) for index in range(10)]
    batch = dset.collate_fn(data)
    # check the nuclear coordinates and numbers match what is expected
    assert batch["pos"].shape[-1] == 3
    assert batch["pos"].ndim == 2
    assert len(batch["atomic_numbers"]) == 10


@pytest.mark.dependency(depends=["test_dataset_load"])
@pytest.mark.local
def test_dgl_dataset(devset_dir):
    dset = MaterialsProjectDataset(
        devset_dir,
        transforms=[transforms.PointCloudToGraphTransform("dgl", cutoff_dist=20.0)],
    )
    for index in range(10):
        data = dset.__getitem__(index)
        assert "graph" in data


@pytest.mark.dependency(depends=["test_dgl_dataset"])
@pytest.mark.local
def test_dgl_collate(devset_dir):
    dset = MaterialsProjectDataset(
        devset_dir,
        transforms=[transforms.PointCloudToGraphTransform("dgl", cutoff_dist=20.0)],
    )
    data = [dset.__getitem__(index) for index in range(10)]
    batch = dset.collate_fn(data)
    assert "graph" in batch
    # should be ten graphs
    assert batch["graph"].batch_size == 10
    assert all([key in batch["graph"].ndata for key in ["pos", "atomic_numbers"]])


@pytest.mark.dependency(depends=["test_dataset_load"])
@pytest.mark.local
def test_dataset_target_keys(devset_dir):
    # this tests target key property without manually grabbing a batch
    dset = MaterialsProjectDataset(devset_dir)
    assert dset.target_keys == {"regression": ["band_gap"]}


def test_saved_devset_pointcloud():
    dset = MaterialsProjectDataset.from_devset()
    samples = [dset.__getitem__(i) for i in range(16)]
    batch = dset.collate_fn(samples)
    assert all([key in batch for key in ["pos", "pc_features", "mask", "targets"]])


def test_saved_devset_graph():
    dset = MaterialsProjectDataset.from_devset(
        transforms=[transforms.PointCloudToGraphTransform("dgl", cutoff_dist=20.0)],
    )
    samples = [dset.__getitem__(i) for i in range(16)]
    batch = dset.collate_fn(samples)
    assert all([key in batch for key in ["graph", "targets"]])
    assert "graph" in batch
    assert all([key in batch["graph"].ndata for key in ["pos", "atomic_numbers"]])
