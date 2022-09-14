# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: MIT License

import pytest

from ocpmodels.datasets import s2ef_devset
from ocpmodels.datasets.base import PointCloudDataset
from ocpmodels.datasets.task_datasets import S2EFDataset


@pytest.fixture(scope="module")
def test_pc_dataset_creation():
    dset = S2EFDataset(s2ef_devset)
    pc_dataset = PointCloudDataset(dset, point_cloud_size=6, sample_size=10)
    return pc_dataset


def test_pc_getitem(test_pc_dataset_creation):
    dset = test_pc_dataset_creation
    dset.__getitem__(0)


def test_pc_collate(test_pc_dataset_creation):
    dset = test_pc_dataset_creation
    batch = [dset.__getitem__(index) for index in range(5)]
    joint_data = dset.collate_fn(batch)
    print(joint_data)
