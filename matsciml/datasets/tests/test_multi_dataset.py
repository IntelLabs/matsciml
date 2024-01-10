from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader

from matsciml.datasets import IS2REDataset, is2re_devset
from matsciml.datasets.materials_project import (
    MaterialsProjectDataset,
    materialsproject_devset,
)
from matsciml.datasets.multi_dataset import MultiDataset

# make the test deterministic
torch.manual_seed(21515)


@pytest.mark.dependency()
def test_joint_dataset():
    is2re = IS2REDataset(is2re_devset)
    mp = MaterialsProjectDataset(materialsproject_devset)

    joint = MultiDataset([is2re, mp])
    # try and grab a sample
    sample = joint.__getitem__(0)


@pytest.mark.dependency(depends=["test_joint_dataset"])
def test_joint_batching():
    is2re = IS2REDataset(is2re_devset)
    mp = MaterialsProjectDataset(materialsproject_devset)

    joint = MultiDataset([is2re, mp])
    # try and grab a sample
    loader = DataLoader(joint, batch_size=8, shuffle=False, collate_fn=joint.collate_fn)
    batch = next(iter(loader))
    assert "IS2REDataset" in batch


@pytest.mark.dependency(depends=["test_joint_batching"])
def test_joint_batching_shuffled():
    is2re = IS2REDataset(is2re_devset)
    mp = MaterialsProjectDataset(materialsproject_devset)

    joint = MultiDataset([is2re, mp])
    # try and grab a sample
    loader = DataLoader(joint, batch_size=8, shuffle=True, collate_fn=joint.collate_fn)
    batch = next(iter(loader))
    # check both datasets are in the batch
    assert all([key in batch for key in ["MaterialsProjectDataset", "IS2REDataset"]])


@pytest.mark.dependency(depends=["test_joint_dataset"])
def test_target_keys():
    is2re = IS2REDataset(is2re_devset)
    mp = MaterialsProjectDataset(materialsproject_devset)

    joint = MultiDataset([is2re, mp])
    keys = joint.target_keys
    expected_is2re = {"regression": ["energy_init", "energy_relaxed"]}
    expected_mp = {"regression": ["band_gap"]}
    assert (
        keys["IS2REDataset"] == expected_is2re
    ), f"IS2REDataset expected {expected_is2re}, got {keys['IS2REDataset']}"
    assert (
        keys["MaterialsProjectDataset"] == expected_mp
    ), f"MaterialsProjectDataset expected {expected_mp}, got {keys['MaterialsProjectDataset']}"
