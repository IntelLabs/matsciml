from hashlib import blake2s

import pytest

from matsciml.datasets import schema

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
