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
        graph.edata["r"] = torch.rand(graph.num_edges(), 1)
        graph.edata["mu"] = torch.rand(graph.num_edges(), 1)
        return {"graph": graph, "graph_variables": torch.rand(1, 16)}

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

    @pytest.mark.dependency()
    def test_mpnn(graph):
        model = MPNN(atom_embedding_dim=128, node_out_dim=16, encoder_only=True)
        with torch.no_grad():
            g_z = model(graph)
        assert g_z.shape == (1, 16)

        # test with grads
        g_z = model(graph)
        assert hasattr(g_z, "grad_fn")
        # make sure every element is finite
        assert torch.isfinite(g_z).all()

    @pytest.mark.dependency()
    def test_schnet_dgl(graph):
        model = SchNet(atom_embedding_dim=128, encoder_only=True)
        with torch.no_grad():
            g_z = model(graph)
        assert g_z.shape == (1, 128)

        # test with grads
        g_z = model(graph)
        assert hasattr(g_z, "grad_fn")
        # make sure every element is finite
        assert torch.isfinite(g_z).all()

    @pytest.mark.dependency()
    def test_egnn_dgl(graph):
        egnn_kwargs = {
            "embed_in_dim": 1,
            "embed_hidden_dim": 32,
            "embed_out_dim": 128,
            "embed_depth": 5,
            "embed_feat_dims": [128, 128, 128],
            "embed_message_dims": [128, 128, 128],
            "embed_position_dims": [64, 64],
            "embed_edge_attributes_dim": 0,
            "embed_activation": "relu",
            "embed_residual": True,
            "embed_normalize": True,
            "embed_tanh": True,
            "embed_activate_last": False,
            "embed_k_linears": 1,
            "embed_use_attention": False,
            "embed_attention_norm": "sigmoid",
            "readout": "sum",
            "node_projection_depth": 3,
            "node_projection_hidden_dim": 128,
            "node_projection_activation": "relu",
            "prediction_out_dim": 1,
            "prediction_depth": 3,
            "prediction_hidden_dim": 128,
            "prediction_activation": "relu",
            "encoder_only": True,
        }

        model = PLEGNNBackbone(**egnn_kwargs)
        with torch.no_grad():
            g_z = model(graph)
        # should match embed_out_dim
        assert g_z.shape == (1, 128)

        # test with grads
        g_z = model(graph)
        assert hasattr(g_z, "grad_fn")
        # make sure every element is finite
        assert torch.isfinite(g_z).all()
