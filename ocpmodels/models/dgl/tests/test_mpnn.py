
import pytest

import dgl
import torch

from ocpmodels.models import MPNN

dgl.seed(215106)


@pytest.fixture
def test_graph():
    num_nodes, num_edges = 50, 80
    graph = dgl.rand_graph(num_nodes, num_edges)
    # generate fake data
    graph.ndata["atomic_numbers"] = torch.randint(1, 100, (num_nodes,))
    graph.edata["r"] = torch.randn((num_edges, 1))
    return graph


def test_mpnn_forward(test_graph):
    model = MPNN(encoder_only=True, node_out_dim=32)
    with torch.no_grad():
        g_z = model(graph=test_graph)
    # 1 graph, node_out_dim
    assert g_z.shape == (1, 32)
    
