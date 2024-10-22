from __future__ import annotations

import pytest
import torch

from matsciml.datasets import S2EFDataset, transforms
from matsciml.lightning.data_utils import MatSciMLDataModule


@pytest.mark.dependency()
def test_distance_transform():
    trans = [
        transforms.PointCloudToGraphTransform(
            "dgl",
            cutoff_dist=20.0,
            node_keys=["pos", "atomic_numbers"],
        ),
        transforms.DistancesTransform(),
    ]
    dset = S2EFDataset.from_devset(transforms=trans)
    batch = dset.__getitem__(0)
    assert "r" in batch.get("graph").edata


@pytest.mark.dependency(["test_distance_transform"])
def test_graph_variable_transform():
    trans = [
        transforms.PointCloudToGraphTransform(
            "dgl",
            cutoff_dist=20.0,
            node_keys=["pos", "atomic_numbers"],
        ),
        transforms.DistancesTransform(),
        transforms.GraphVariablesTransform(),
    ]
    dset = S2EFDataset.from_devset(transforms=trans)
    batch = dset.__getitem__(0)
    assert "graph_variables" in batch


@pytest.mark.dependency(["test_graph_variable_transform"])
def test_batched_gv_transform():
    trans = [
        transforms.PointCloudToGraphTransform(
            "dgl",
            cutoff_dist=20.0,
            node_keys=["pos", "atomic_numbers"],
        ),
        transforms.DistancesTransform(),
        transforms.GraphVariablesTransform(),
    ]
    dm = MatSciMLDataModule.from_devset(
        "S2EFDataset",
        dset_kwargs={"transforms": trans},
    )
    dm.setup()
    loader = dm.train_dataloader()
    batch = next(iter(loader))
    assert "graph_variables" in batch
    gv = batch.get("graph_variables")
    assert gv.ndim == 2
    assert gv.shape == (8, 9)
    assert torch.all(~torch.isnan(gv))


@pytest.mark.dependency()
def test_remove_tag_zero():
    trans = [
        transforms.PointCloudToGraphTransform(
            "dgl",
            cutoff_dist=20.0,
            node_keys=["pos", "atomic_numbers", "tags"],
        ),
        transforms.RemoveTagZeroNodes(),
    ]
    dm = MatSciMLDataModule.from_devset(
        "S2EFDataset",
        dset_kwargs={"transforms": trans},
    )
    dm.setup()
    loader = dm.train_dataloader()
    graph = next(iter(loader))["graph"]
    # make sure we've purged all of the tag zero nodes
    assert not torch.any(graph.ndata["tags"] == 0)


@pytest.mark.dependency(["test_remove_tag_zero"])
def test_graph_supernode():
    trans = [
        transforms.PointCloudToGraphTransform(
            "dgl",
            cutoff_dist=20.0,
            node_keys=["pos", "atomic_numbers", "tags", "fixed"],
        ),
        transforms.GraphSuperNodes(100),
        transforms.RemoveTagZeroNodes(),
    ]
    dm = MatSciMLDataModule.from_devset(
        "S2EFDataset",
        dset_kwargs={"transforms": trans},
    )
    dm.setup()
    loader = dm.train_dataloader()
    graph = next(iter(loader))["graph"]
    # should be one super node per graph
    assert (graph.ndata["tags"] == 3).sum() == graph.batch_size


@pytest.mark.dependency(["test_remove_tag_zero"])
def test_atom_supernode():
    trans = [
        transforms.PointCloudToGraphTransform(
            "dgl",
            cutoff_dist=20.0,
            node_keys=["pos", "atomic_numbers", "tags", "fixed"],
        ),
        transforms.AtomicSuperNodes(100),
        transforms.RemoveTagZeroNodes(),
    ]
    dm = MatSciMLDataModule.from_devset(
        "S2EFDataset",
        dset_kwargs={"transforms": trans},
    )
    dm.setup()
    loader = dm.train_dataloader()
    graph = next(iter(loader))["graph"]
    # make sure node numbers don't exceed the expected limit
    assert torch.all(graph.ndata["atomic_numbers"] <= 199)
    # make sure we have atomic super nodes after the transform
    assert torch.any(graph.ndata["tags"] == 4)


@pytest.mark.dependency(["test_atom_supernode", "test_graph_supernode"])
def test_all_supernodes():
    trans = [
        transforms.PointCloudToGraphTransform(
            "dgl",
            cutoff_dist=20.0,
            node_keys=["pos", "atomic_numbers", "tags", "fixed"],
        ),
        transforms.GraphSuperNodes(100),
        transforms.AtomicSuperNodes(100),
        transforms.RemoveTagZeroNodes(),
    ]
    dm = MatSciMLDataModule.from_devset(
        "S2EFDataset",
        dset_kwargs={"transforms": trans},
    )
    dm.setup()
    loader = dm.train_dataloader()
    graph = next(iter(loader))["graph"]
    # make sure node numbers don't exceed the expected limit
    assert torch.all(graph.ndata["atomic_numbers"] <= 200)
    # make sure we have graph and atomic super nodes after the transform
    assert torch.any(graph.ndata["tags"] == 4)
    assert torch.any(graph.ndata["tags"] == 3)
    # check no tag zero nodes remain
    assert not torch.any(graph.ndata["tags"] == 0)
    # make sure the graph super node has an embedding index of 100
    mask = graph.ndata["tags"] == 3
    assert (graph.ndata["atomic_numbers"][mask] - 100).sum() == 0


@pytest.mark.skip(reason="Broken test.")
def test_graph_sorting():
    trans = [
        transforms.PointCloudToGraphTransform(
            "dgl",
            cutoff_dist=20.0,
            node_keys=["pos", "atomic_numbers"],
        ),
        transforms.GraphReordering("metis", k=10),
    ]
    dm = MatSciMLDataModule.from_devset(
        "S2EFDataset",
        dset_kwargs={"transforms": trans},
    )
    dm.setup()
    loader = dm.train_dataloader()
    _ = next(iter(loader))["graph"]
    # not really anything to test, but just make sure it runs :D


@pytest.mark.parametrize("backend", ["ase", "pymatgen"])
def test_ase_periodic(backend):
    trans = [
        transforms.PeriodicPropertiesTransform(
            cutoff_radius=6.0, adaptive_cutoff=True, backend=backend
        )
    ]
    dm = MatSciMLDataModule.from_devset(
        "S2EFDataset",
        dset_kwargs={"transforms": trans},
    )
    dm.setup()
    loader = dm.train_dataloader()
    batch = next(iter(loader))
    # check if periodic properties transform was applied
    assert "unit_offsets" in batch


def test_pbc_backend_equivalence_easy():
    from ase.build import molecule
    from pymatgen.io.ase import AseAtomsAdaptor

    atoms = molecule(
        "H2O", cell=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], pbc=(True, True, True)
    )
    structure = AseAtomsAdaptor.get_structure(atoms)

    data = {}
    coords = torch.from_numpy(structure.cart_coords).float()
    data["pos"] = coords
    atom_numbers = torch.LongTensor(structure.atomic_numbers)
    data["atomic_numbers"] = atom_numbers
    data["natoms"] = len(atom_numbers)
    lattice_params = torch.FloatTensor(
        structure.lattice.abc
        + tuple(a * (torch.pi / 180.0) for a in structure.lattice.angles),
    )
    lattice_features = {
        "lattice_params": lattice_params,
    }
    data["lattice_features"] = lattice_features

    ase_trans = transforms.PeriodicPropertiesTransform(
        cutoff_radius=6.0, adaptive_cutoff=True, backend="ase"
    )

    pymatgen_trans = transforms.PeriodicPropertiesTransform(
        cutoff_radius=6.0, adaptive_cutoff=True, backend="pymatgen"
    )

    ase_result = ase_trans(data)
    pymatgen_result = pymatgen_trans(data)

    ase_wiring = torch.vstack([ase_result["src_nodes"], ase_result["dst_nodes"]])
    pymatgen_wiring = torch.vstack(
        [pymatgen_result["src_nodes"], pymatgen_result["dst_nodes"]]
    )
    equivalence = ase_wiring == pymatgen_wiring
    # basically checking if src -> dst node wiring is equivalent between the two approaches
    assert torch.all(equivalence)


def test_pbc_backend_equivalence_hard():
    ase_trans = transforms.PeriodicPropertiesTransform(
        cutoff_radius=6.0, adaptive_cutoff=True, backend="ase"
    )

    pymatgen_trans = transforms.PeriodicPropertiesTransform(
        cutoff_radius=6.0, adaptive_cutoff=True, backend="pymatgen"
    )

    dm = MatSciMLDataModule.from_devset(
        "S2EFDataset",
        batch_size=1,
    )

    dm.setup()
    loader = dm.train_dataloader()
    batch = next(iter(loader))
    batch["atomic_numbers"] = batch["atomic_numbers"].squeeze(0)

    ase_result = ase_trans(batch)
    pymatgen_result = pymatgen_trans(batch)
    ase_wiring = torch.vstack([ase_result["src_nodes"], ase_result["dst_nodes"]])
    pymatgen_wiring = torch.vstack(
        [pymatgen_result["src_nodes"], pymatgen_result["dst_nodes"]]
    )
    equivalence = ase_wiring == pymatgen_wiring
    # basically checking if src -> dst node wiring is equivalent between the two approaches
    assert torch.all(equivalence)
