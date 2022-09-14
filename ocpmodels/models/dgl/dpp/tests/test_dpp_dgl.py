# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: MIT License

import torch
import dgl
from ocpmodels.models import DimeNetPP

torch.random.manual_seed(10)
dgl.random.seed(10)


@torch.no_grad()
def test_dimenet_pp_single():
    # use the default settings
    model = DimeNetPP()
    # construct a random graph
    single_graph = dgl.rand_graph(10, 50)
    single_graph = dgl.remove_self_loop(single_graph)
    # generate positions and atomic numbers
    single_graph.ndata["pos"] = torch.rand(10, 3)
    single_graph.ndata["atomic_numbers"] = torch.randint(0, 100, (10,)).long()
    # test distance computation
    single_graph = model.edge_distance(single_graph)
    # now test the full computation unit testing lol
    output = model(single_graph)
    assert torch.isfinite(output).all()


@torch.no_grad()
def test_dimenet_pp_batch():
    # use the default settings
    model = DimeNetPP()
    # construct a random graph
    single_graph = dgl.rand_graph(10, 50)
    single_graph = dgl.remove_self_loop(single_graph)
    # generate positions and atomic numbers
    single_graph.ndata["pos"] = torch.rand(10, 3)
    single_graph.ndata["atomic_numbers"] = torch.randint(0, 100, (10,)).long()
    graphs = dgl.batch(
        [
            single_graph,
        ]
        * 10
    )
    # test distance computation
    graphs = model.edge_distance(graphs)
    # now test the full computation unit testing lol
    output = model(graphs)
    assert output.shape == (graphs.batch_size, 1)
    assert torch.isfinite(output).all()
