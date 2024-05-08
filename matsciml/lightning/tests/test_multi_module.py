from __future__ import annotations

import pytest

from matsciml.datasets import IS2REDataset, S2EFDataset
from matsciml.datasets.materials_project import MaterialsProjectDataset
from matsciml.datasets.multi_dataset import MultiDataset
from matsciml.lightning.data_utils import MultiDataModule


@pytest.fixture
def datamodule():
    dset = MultiDataset(
        [
            S2EFDataset.from_devset(),
            IS2REDataset.from_devset(),
            MaterialsProjectDataset.from_devset(),
        ],
    )
    dm = MultiDataModule(train_dataset=dset, batch_size=8)
    return dm


@pytest.mark.dependency()
def test_setup(datamodule):
    assert getattr(datamodule, "datasets", False)


@pytest.mark.dependency(depends=["test_setup"])
def test_target_keys(datamodule):
    keys = datamodule.target_keys
    expected = {
        "S2EFDataset": {"regression": ["energy", "force"]},
        "IS2REDataset": {"regression": ["energy_init", "energy_relaxed"]},
        "MaterialsProjectDataset": {
            "classification": ["is_metal", "is_magnetic", "is_stable"],
            "regression": [
                "uncorrected_energy_per_atom",
                "efermi",
                "energy_per_atom",
                "band_gap",
                "formation_energy_per_atom",
            ],
        },
    }
    assert keys.keys() == expected.keys()
    for key, target_values in expected.items():
        assert sorted(target_values) == sorted(
            keys[key]
        ), f"Expected target keys {target_values}, got {keys[key]}"
