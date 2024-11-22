from hashlib import blake2s
from datetime import datetime

import pytest

from matsciml.datasets import schema
from matsciml.datasets.transforms import PeriodicPropertiesTransform

fake_hashes = {
    key: blake2s(bytes(key.encode("utf-8"))).hexdigest()
    for key in ["train", "test", "validation", "predict"]
}


def test_split_schema_pass():
    s = schema.SplitHashSchema(**fake_hashes)
    assert s


def test_partial_schema_pass():
    s = schema.SplitHashSchema(
        train=fake_hashes["train"], validation=fake_hashes["validation"]
    )
    assert s


def test_bad_hash_fail():
    with pytest.raises(ValueError):
        # chop off the end of the hash
        s = schema.SplitHashSchema(train=fake_hashes["train"][:-2])  # noqa: F841


def test_no_hashes():
    with pytest.raises(RuntimeError):
        s = schema.SplitHashSchema()  # noqa: F841


def test_dataset_minimal_schema_pass():
    splits = schema.SplitHashSchema(
        train=fake_hashes["train"], validation=fake_hashes["validation"]
    )
    dset = schema.DatasetSchema(
        name="GenericDataset",
        creation=datetime.now(),
        dataset_type="SCFCycle",
        target_keys=["energy", "forces"],
        split_blake2s=splits,
    )
    assert dset


def test_dataset_minimal_schema_roundtrip():
    """Make sure the dataset minimal schema can dump and reload"""
    splits = schema.SplitHashSchema(
        train=fake_hashes["train"], validation=fake_hashes["validation"]
    )
    dset = schema.DatasetSchema(
        name="GenericDataset",
        creation=datetime.now(),
        dataset_type="SCFCycle",
        target_keys=["energy", "forces"],
        split_blake2s=splits,
    )
    json_rep = dset.model_dump_json()
    reloaded_dset = dset.model_validate_json(json_rep)
    assert reloaded_dset == dset


@pytest.mark.parametrize("backend", ["pymatgen", "ase"])
@pytest.mark.parametrize("cutoff_radius", [3.0, 6.0, 10.0])
def test_graph_wiring_from_transform(backend, cutoff_radius):
    transform = PeriodicPropertiesTransform(cutoff_radius, backend=backend)
    s = schema.GraphWiringSchema.from_transform(transform, allow_mismatch=False)
    recreate = s.to_transform()
    assert transform.__dict__ == recreate.__dict__
