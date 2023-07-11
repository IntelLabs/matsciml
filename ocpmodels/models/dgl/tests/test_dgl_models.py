import pytest

from ocpmodels.common import package_registry

if package_registry["dgl"]:
    import dgl
    import torch

    from ocpmodels.models.dgl import *

    dgl.seed(2151)
    torch.manual_seed(21610)

    @pytest.fixture
    def graph():
        graph = dgl.rand_graph(15, 20)
        graph = dgl.add_self_loop(graph)
        graph.ndata["pos"] = torch.rand(15, 3)
        graph.ndata["atomic_numbers"] = torch.randint(0, 100, (15,))
        return {"graph": graph}

    @pytest.mark.dependency()
    def test_gcn_conv(graph):
        model = GraphConvModel(
            atom_embedding_dim=128, out_dim=64, num_blocks=3, encoder_only=True
        )
        # test without grads
        with torch.no_grad():
            g_z = model(graph)
        assert g_z.shape == (1, 64)

        # test with grads
        g_z = model(graph)
        assert hasattr(g_z, "grad_fn")
        # make sure every element is finite
        assert torch.isfinite(g_z).all()
