from hashlib import blake2s
from datetime import datetime

import pytest
import numpy as np
import torch
from ase.geometry import cell_to_cellpar

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
        targets=[
            {
                "name": "total_energy",
                "shape": "0",
                "description": "Total energy of the system.",
            },
            {
                "name": "forces",
                "shape": "*,3",
                "description": "Atomic forces per node.",
            },
        ],
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
        targets=[
            {
                "name": "total_energy",
                "shape": "0",
                "description": "Total energy of the system.",
            },
            {
                "name": "forces",
                "shape": "*,3",
                "description": "Atomic forces per node.",
            },
        ],
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


@pytest.mark.parametrize("backend", ["pymatgen", "ase"])
def test_graph_wiring_version_mismatch(backend):
    """Ensures that an exception is thrown when backend version does not match"""
    with pytest.raises(RuntimeError):
        s = schema.GraphWiringSchema(  # noqa: F841
            cutoff_radius=10.0,
            algorithm=backend,
            allow_mismatch=False,
            algo_version="fake_version",
            adaptive_cutoff=False,
        )


@pytest.mark.parametrize("num_atoms", [5, 12, 16])
@pytest.mark.parametrize("array_lib", ["numpy", "torch"])
def test_basic_data_sample_schema(num_atoms, array_lib):
    if array_lib == "numpy":
        coords = np.random.rand(num_atoms, 3)
        numbers = np.random.randint(1, 100, (num_atoms))
    else:
        coords = torch.rand(num_atoms, 3)
        numbers = torch.randint(1, 100, (num_atoms,))
    pbc = {"x": True, "y": True, "z": True}
    data = schema.DataSampleSchema(
        index=0,
        num_atoms=num_atoms,
        cart_coords=coords,
        atomic_numbers=numbers,
        pbc=pbc,
        datatype="SCFCycle",
    )
    assert data


@pytest.mark.parametrize("num_atoms", [5, 12, 16])
@pytest.mark.parametrize("array_lib", ["numpy", "torch"])
def test_basic_data_sample_roundtrip(num_atoms, array_lib):
    if array_lib == "numpy":
        coords = np.random.rand(num_atoms, 3)
        numbers = np.random.randint(1, 100, (num_atoms))
    else:
        coords = torch.rand(num_atoms, 3)
        numbers = torch.randint(1, 100, (num_atoms,))
    pbc = {"x": True, "y": True, "z": True}
    data = schema.DataSampleSchema(
        index=0,
        num_atoms=num_atoms,
        cart_coords=coords,
        atomic_numbers=numbers,
        pbc=pbc,
        datatype="SCFCycle",
    )
    json = data.model_dump_json()
    recreate = schema.DataSampleSchema.model_validate_json(json)
    assert recreate == data


@pytest.mark.parametrize("num_atoms", [3, 10, 25])
@pytest.mark.parametrize("array_lib", ["numpy", "torch"])
def test_data_sample_fail_coord_shape(num_atoms, array_lib):
    if array_lib == "numpy":
        coords = np.random.rand(num_atoms, 5)
        numbers = np.random.randint(1, 100, (num_atoms))
    else:
        coords = torch.rand(num_atoms, 5)
        numbers = torch.randint(1, 100, (num_atoms,))
    pbc = {"x": True, "y": True, "z": True}
    with pytest.raises(ValueError):
        data = schema.DataSampleSchema(  # noqa: F841
            index=0,
            num_atoms=num_atoms,
            cart_coords=coords,
            atomic_numbers=numbers,
            pbc=pbc,
            datatype="SCFCycle",
        )


def test_lattice_param_to_matrix_consistency():
    """Make sure that lattice parameters map to matrix correctly during validation"""
    coords = np.random.rand(5, 3)
    numbers = np.random.randint(1, 100, (5))
    data = schema.DataSampleSchema(
        index=0,
        num_atoms=5,
        cart_coords=coords,
        atomic_numbers=numbers,
        pbc={"x": True, "y": True, "z": True},
        datatype="OptimizationCycle",
        lattice_parameters=[5.0, 5.0, 5.0, 90.0, 90.0, 90.0],
    )
    assert data.frac_coords is not None
    assert data.lattice_matrix is not None
    reconverted = cell_to_cellpar(data.lattice_matrix)
    assert np.allclose(reconverted, data.lattice_parameters)
    exact = np.array([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]])
    assert np.allclose(exact, data.lattice_matrix)
