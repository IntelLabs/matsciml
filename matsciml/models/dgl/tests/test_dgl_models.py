from __future__ import annotations

import pytest

from matsciml.common import package_registry

if package_registry["dgl"]:
    import dgl
    import torch

    from matsciml.models.dgl import *

    dgl.seed(2151)
    torch.manual_seed(21610)

    @pytest.fixture
    def graph():
        graph = dgl.graph(
            [[0, 1], [1, 2], [2, 3], [3, 0], [3, 4], [4, 5], [4, 6], [4, 7]],
        )
        graph.ndata["pos"] = torch.rand(graph.num_nodes(), 3)
        graph.ndata["atomic_numbers"] = torch.randint(0, 100, (graph.num_nodes(),))
        graph.edata["r"] = torch.rand(graph.num_edges(), 1)
        graph.edata["mu"] = torch.rand(
            graph.num_edges(),
        )
        return {"graph": graph, "graph_variables": torch.rand(1, 16)}

    @pytest.mark.dependency()
    def test_gcn_conv(graph):
        model = GraphConvModel(
            atom_embedding_dim=128,
            out_dim=64,
            num_blocks=3,
            encoder_only=True,
        )
        # test without grads
        with torch.no_grad():
            g_z = model(graph)
        assert g_z.system_embedding.shape == (1, 64)
        # test with grads
        g_z = model(graph)
        assert hasattr(g_z.system_embedding, "grad_fn")
        # make sure every element is finite
        assert torch.isfinite(g_z.system_embedding).all()

    @pytest.mark.dependency()
    def test_mpnn(graph):
        model = MPNN(atom_embedding_dim=128, node_out_dim=16, encoder_only=True)
        with torch.no_grad():
            g_z = model(graph)
        assert g_z.system_embedding.shape == (1, 16)

        # test with grads
        g_z = model(graph)
        assert hasattr(g_z.system_embedding, "grad_fn")
        # make sure every element is finite
        assert torch.isfinite(g_z.system_embedding).all()

    @pytest.mark.dependency()
    def test_schnet_dgl(graph):
        model = SchNet(atom_embedding_dim=128, encoder_only=True)
        with torch.no_grad():
            g_z = model(graph)
        assert g_z.system_embedding.shape == (1, 128)

        # test with grads
        g_z = model(graph)
        assert hasattr(g_z.system_embedding, "grad_fn")
        # make sure every element is finite
        assert torch.isfinite(g_z.system_embedding).all()

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
        assert g_z.system_embedding.shape == (1, 128)

        # test with grads
        g_z = model(graph)
        assert hasattr(g_z.system_embedding, "grad_fn")
        # make sure every element is finite
        assert torch.isfinite(g_z.system_embedding).all()

    @pytest.mark.dependency()
    def test_megnet_dgl(graph):
        megnet_kwargs = {
            "edge_feat_dim": 2,
            "node_feat_dim": 64,
            "graph_feat_dim": 16,
            "num_blocks": 2,
            "hiddens": [64, 64],
            "conv_hiddens": [64, 64],
            "s2s_num_layers": 2,
            "s2s_num_iters": 1,
            "output_hiddens": [64, 64],
            "is_classification": False,
        }

        model = MEGNet(**megnet_kwargs)
        with torch.no_grad():
            g_z = model(graph)
        # should match 128 + 128 + 64
        assert g_z.system_embedding.shape == (1, 320)

        # test with grads
        g_z = model(graph)
        assert hasattr(g_z.system_embedding, "grad_fn")
        # make sure every element is finite
        assert torch.isfinite(g_z.system_embedding).all()

    @pytest.mark.dependency()
    def test_dpp_dgl(graph):
        model = DimeNetPP()
        with torch.no_grad():
            g_z = model(graph)
        # should match the 'out_emb_size' argument
        assert g_z.system_embedding.shape == (1, 256)

        # test with grads
        g_z = model(graph)
        assert hasattr(g_z.system_embedding, "grad_fn")
        # make sure every element is finite
        assert torch.isfinite(g_z.system_embedding).all()


@pytest.mark.dependency()
def test_m3gnet_dgl(graph):
    import numpy as np
    from matgl.graph.compute import compute_pair_vector_and_distance

    graph["graph"].ndata["node_type"] = graph["graph"].ndata["atomic_numbers"]
    graph["graph"].ndata["num_nodes"] = torch.Tensor(
        len(graph["graph"].ndata["node_type"]),
    )
    images = np.zeros((len(graph["graph"].edges()[0]), 3))
    lattice_matrix = np.zeros((1, 3, 3))
    pbc_offset = torch.tensor(images, dtype=torch.float64)
    graph["graph"].edata["pbc_offset"] = pbc_offset.to(torch.int32)
    graph["graph"].edata["pbc_offshift"] = torch.matmul(
        pbc_offset,
        torch.tensor(lattice_matrix[0]),
    )
    graph["graph"].edata["lattice"] = torch.tensor(
        np.repeat(lattice_matrix, graph["graph"].num_edges(), axis=0),
        dtype=torch.float32,
    )
    ######

    bond_vec, bond_dist = compute_pair_vector_and_distance(graph["graph"])
    graph["graph"].edata["bond_vec"] = bond_vec
    graph["graph"].edata["bond_dist"] = bond_dist
    # fmt: off
    element_types = [
        'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na',
        'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc',
        'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga',
        'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb',
        'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb',
        'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm',
        'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
        'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl',
        'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa',
        'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md',
        'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg',
        'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og',
    ]
    # fmt: on

    model = M3GNet(element_types=element_types)
    with torch.no_grad():
        g_z = model(graph)
    # Scalar output right now
    assert g_z.system_embedding.shape == torch.Size([1, 64])

    # test with grads
    g_z = model(graph)
    assert hasattr(g_z.system_embedding, "grad_fn")
    # make sure every element is finite
    assert torch.isfinite(g_z.system_embedding).all()
