from __future__ import annotations

import pytest

from matsciml.common.registry import registry
from matsciml.datasets.transforms import DummyTransform

dsets = list(registry.__entries__["datasets"].values())

"""
Unit tests for testing the dataset preprocessing dumping and loading
functionality. Note that this might cause `/tmp` to grow if left
unmonitored!
"""


@pytest.mark.parametrize("dset", dsets)
def test_dataset_serial_save(dset, tmp_path):
    if getattr(dset, "__devset__", None):
        devset = dset.from_devset()
        assert not devset.is_preprocessed
        devset.save_preprocessed_data(tmp_path, num_procs=1)


@pytest.mark.parametrize("dset", dsets)
def test_dataset_parallel_save(dset, tmp_path):
    if getattr(dset, "__devset__", None):
        devset = dset.from_devset()
        assert not devset.is_preprocessed
        devset.save_preprocessed_data(tmp_path, num_procs=4)


@pytest.mark.parametrize("dset", dsets)
def test_dataset_cycle(dset, tmp_path):
    if getattr(dset, "__devset__", None):
        devset = dset.from_devset(transforms=[DummyTransform()])
        assert not devset.is_preprocessed
        devset.save_preprocessed_data(tmp_path, num_procs=2)
        new_devset = dset(tmp_path)
        assert new_devset.is_preprocessed
        sample = new_devset.__getitem__(0)
        assert sample["touched"]
