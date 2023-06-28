import pytest

import torch

from ocpmodels.datasets import is2re_devset, IS2REDataset
from ocpmodels.datasets.transforms import GraphToPointCloudTransform
from ocpmodels.common import package_registry


if package_registry["dgl"]:
    import dgl

    @pytest.fixture()
    def demo_dgl_graph():
        g = dgl.rand_graph(10, 15)
        g.ndata["pos"] = torch.rand(10, 3)
        g.ndata["atomic_numbers"] = torch.randint(1, 100, (10,))
        data = {
            "graph": g,
            "node_feats": torch.rand(10, 5),
            "edge_feats": torch.rand(15, 2),
            "dataset": "IS2REDataset",
        }
        return data

    @pytest.mark.dependency()
    def test_transform_init():
        t = GraphToPointCloudTransform("dgl", atom_centered=True)

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
        depends=["test_transform_init", "test_dgl_atom_center_transform"]
    )
    def test_dgl_is2re():
        dset = IS2REDataset(
            is2re_devset,
            transforms=[GraphToPointCloudTransform("dgl", atom_centered=True)],
        )
        sample = dset.__getitem__(0)
        assert "pc_features" in sample
        assert "pos" in sample
        # make sure positions are atom centered
        assert sample["pos"].ndim == 3
