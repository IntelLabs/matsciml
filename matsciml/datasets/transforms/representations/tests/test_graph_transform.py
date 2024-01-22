from __future__ import annotations

import pytest
import torch

from matsciml.common import package_registry
from matsciml.datasets import IS2REDataset, S2EFDataset, is2re_devset, s2ef_devset
from matsciml.datasets.materials_project import (
    MaterialsProjectDataset,
    materialsproject_devset,
)
from matsciml.datasets.transforms import (
    GraphToPointCloudTransform,
    PointCloudToGraphTransform,
)

if package_registry["dgl"]:
    import dgl

    @pytest.fixture()
    def demo_dgl_graph():
        g = dgl.graph(
            [[0, 1], [1, 2], [2, 3], [2, 4], [2, 5], [3, 6], [6, 7], [6, 8], [6, 9]],
        )
        g.ndata["pos"] = torch.rand(g.num_nodes(), 3)
        g.ndata["atomic_numbers"] = torch.randint(1, 100, (g.num_nodes(),))
        g = dgl.to_bidirected(g, copy_ndata=True)
        data = {
            "graph": g,
            "node_feats": torch.rand(10, 5),
            "edge_feats": torch.rand(15, 2),
            "dataset": "IS2REDataset",
        }
        return data

    @pytest.mark.dependency()
    def test_transform_init():
        t = GraphToPointCloudTransform("dgl", full_pairwise=True)

    @pytest.mark.dependency(depends=["test_transform_init"])
    def test_dgl_atom_center_transform(demo_dgl_graph):
        data = demo_dgl_graph
        t = GraphToPointCloudTransform("dgl")
        data = t(data)
        assert all([key in data for key in ["pc_features", "pos", "dataset"]])

    @pytest.mark.dependency(depends=["test_transform_init"])
    def test_dgl_transform_fail(demo_dgl_graph):
        data = demo_dgl_graph
        t = GraphToPointCloudTransform("dgl")
        del data["graph"]
        with pytest.raises(AssertionError):
            t(data)

    @pytest.mark.dependency(
        depends=["test_transform_init", "test_dgl_atom_center_transform"],
    )
    def test_dgl_pairwise_is2re():
        dset = IS2REDataset(
            is2re_devset,
            transforms=[
                PointCloudToGraphTransform(
                    "dgl",
                    cutoff_dist=20.0,
                    node_keys=["pos", "atomic_numbers"],
                ),
            ],
        )
        sample = dset.__getitem__(0)
        assert "graph" in sample
        # make sure positions are still same dimension
        assert sample["graph"].ndata["pos"].ndim == 2

    @pytest.mark.dependency(
        depends=["test_transform_init", "test_dgl_atom_center_transform"],
    )
    def test_dgl_pairwise_s2ef():
        dset = S2EFDataset(
            s2ef_devset,
            transforms=[
                PointCloudToGraphTransform(
                    "dgl",
                    cutoff_dist=20.0,
                    node_keys=["pos", "atomic_numbers"],
                ),
            ],
        )
        sample = dset.__getitem__(0)
        assert "graph" in sample
        # make sure positions are still same dimension
        assert sample["graph"].ndata["pos"].ndim == 2

    @pytest.mark.dependency(
        depends=["test_transform_init", "test_dgl_atom_center_transform"],
    )
    def test_dgl_materials_project_fail():
        # makes sure this cannot be applied to a dataset with point clouds already
        dset = MaterialsProjectDataset(
            materialsproject_devset,
            transforms=[GraphToPointCloudTransform("dgl", full_pairwise=True)],
        )
        with pytest.raises(
            AssertionError,
            match="No graphs to transform into point clouds!",
        ):
            sample = dset.__getitem__(0)
