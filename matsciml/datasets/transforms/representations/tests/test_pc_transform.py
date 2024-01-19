from __future__ import annotations

import pytest
import torch

from matsciml.common import package_registry
from matsciml.datasets import IS2REDataset, is2re_devset
from matsciml.datasets.lips import LiPSDataset, lips_devset
from matsciml.datasets.materials_project import (
    MaterialsProjectDataset,
    materialsproject_devset,
)
from matsciml.datasets.symmetry import SyntheticPointGroupDataset, symmetry_devset
from matsciml.datasets.transforms import PointCloudToGraphTransform

if package_registry["dgl"]:
    import dgl

    @pytest.fixture()
    def pc_data():
        data = {
            "node_feats": torch.rand(10, 5),
            "edge_feats": torch.rand(15, 2),
            "atomic_numbers": torch.randint(1, 100, (10,)),
            "pos": torch.rand(10, 3),
            "dataset": "FakeDataset",
        }
        return data

    @pytest.mark.dependency()
    def test_transform_init():
        t = PointCloudToGraphTransform("dgl")

    @pytest.mark.dependency(depends=["test_transform_init"])
    def test_dgl_create(pc_data):
        t = PointCloudToGraphTransform("dgl")
        data = t(pc_data)
        assert all([key in data for key in ["graph", "dataset"]])

    @pytest.mark.dependency(depends=["test_dgl_create"])
    def test_dgl_data_copy(pc_data):
        t = PointCloudToGraphTransform(
            "dgl",
            node_keys=["pos", "atomic_numbers", "node_feats"],
        )
        data = t(pc_data)
        graph = data.get("graph")
        assert all([key in data for key in ["graph", "dataset"]])
        assert all(
            [key in graph.ndata for key in ["pos", "atomic_numbers", "node_feats"]],
        )

    @pytest.mark.dependency(depends=["test_transform_init"])
    def test_dgl_transform_fail(pc_data):
        t = PointCloudToGraphTransform("dgl")
        del pc_data["pos"]
        with pytest.raises(AssertionError):
            t(pc_data)

    @pytest.mark.dependency(depends=["test_transform_init", "test_dgl_create"])
    def test_dgl_materials_project():
        dset = MaterialsProjectDataset(
            materialsproject_devset,
            transforms=[PointCloudToGraphTransform("dgl")],
        )
        sample = dset.__getitem__(0)
        assert "graph" in sample.keys()
        g = sample.get("graph")
        assert all([key in g.ndata for key in ["pos", "atomic_numbers"]])

    @pytest.mark.dependency(depends=["test_transform_init", "test_dgl_create"])
    def test_dgl_lips():
        dset = LiPSDataset(lips_devset, transforms=[PointCloudToGraphTransform("dgl")])
        sample = dset.__getitem__(0)
        assert "graph" in sample.keys()
        g = sample.get("graph")
        assert all([key in g.ndata for key in ["pos", "atomic_numbers", "force"]])

    @pytest.mark.dependency(depends=["test_transform_init", "test_dgl_create"])
    def test_dgl_symmetry():
        dset = SyntheticPointGroupDataset(
            symmetry_devset,
            transforms=[PointCloudToGraphTransform("dgl")],
        )
        sample = dset.__getitem__(0)
        assert "graph" in sample.keys()
        g = sample.get("graph")
        assert all([key in g.ndata for key in ["pos", "atomic_numbers"]])


if package_registry["pyg"]:

    @pytest.mark.dependency()
    def test_transform_pyg_init():
        t = PointCloudToGraphTransform("pyg")

    @pytest.mark.dependency(depends=["test_transform_pyg_init"])
    def test_pyg_create(pc_data):
        t = PointCloudToGraphTransform("pyg")
        data = t(pc_data)
        assert all([key in data for key in ["graph", "dataset"]])
        assert all(
            [getattr(data["graph"], key).sum() for key in ["pos", "atomic_numbers"]],
        )

    @pytest.mark.dependency(depends=["test_pyg_create"])
    def test_pyg_data_copy(pc_data):
        t = PointCloudToGraphTransform(
            "pyg",
            node_keys=["pos", "atomic_numbers", "node_feats"],
        )
        data = t(pc_data)
        graph = data.get("graph")
        assert all([key in data for key in ["graph", "dataset"]])
        assert all([key in graph for key in ["pos", "atomic_numbers", "node_feats"]])

    @pytest.mark.dependency(depends=["test_transform_pyg_init", "test_pyg_create"])
    def test_pyg_materials_project():
        dset = MaterialsProjectDataset(
            materialsproject_devset,
            transforms=[PointCloudToGraphTransform("pyg")],
        )
        sample = dset.__getitem__(0)
        assert "graph" in sample.keys()
        g = sample.get("graph")
        assert all([key in g for key in ["pos", "atomic_numbers"]])

    @pytest.mark.dependency(depends=["test_transform_pyg_init", "test_pyg_create"])
    def test_pyg_lips():
        dset = LiPSDataset(
            lips_devset,
            transforms=[PointCloudToGraphTransform("pyg", node_keys=["force"])],
        )
        sample = dset.__getitem__(0)
        assert "graph" in sample.keys()
        g = sample.get("graph")
        assert all([key in g for key in ["pos", "atomic_numbers", "force"]])

    @pytest.mark.dependency(depends=["test_transform_pyg_init", "test_pyg_create"])
    def test_pyg_symmetry():
        dset = SyntheticPointGroupDataset(
            symmetry_devset,
            transforms=[PointCloudToGraphTransform("pyg")],
        )
        # TODO output sample only contains 'coordinates' and nothing similar to 'atomic numbers'
        sample = dset.__getitem__(0)
        assert "graph" in sample.keys()
        g = sample.get("graph")
        assert all([key in g for key in ["pos", "atomic_numbers"]])
