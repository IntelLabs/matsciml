from __future__ import annotations

from typing import Union

import dgl
import torch
from matgl.graph.compute import (
    compute_pair_vector_and_distance,
    compute_theta_and_phi,
    create_line_graph,
)
from matgl.models import M3GNet
from matgl.models._megnet import *
from matgl.utils.cutoff import polynomial_cutoff

from matsciml.common.types import Embeddings


def forward(
    self,
    g: dgl.DGLGraph,
    state_attr: torch.Tensor | None = None,
    l_g: dgl.DGLGraph | None = None,
):
    """Performs message passing and updates node representations.

    Args:
        g : DGLGraph for a batch of graphs.
        state_attr: State attrs for a batch of graphs.
        l_g : DGLGraph for a batch of line graphs.

    Returns:
        output: Output property for a batch of graphs
    """
    g = g["graph"]
    node_types = g.ndata["node_type"]
    bond_vec, bond_dist = compute_pair_vector_and_distance(g)
    g.edata["bond_vec"] = bond_vec
    g.edata["bond_dist"] = bond_dist

    expanded_dists = self.bond_expansion(g.edata["bond_dist"])
    if l_g is None:
        l_g = create_line_graph(g, self.threebody_cutoff)
    else:
        valid_three_body = g.edata["bond_dist"] <= self.threebody_cutoff
        if l_g.num_nodes() == g.edata["bond_vec"][valid_three_body].shape[0]:
            l_g.ndata["bond_vec"] = g.edata["bond_vec"][valid_three_body]
            l_g.ndata["bond_dist"] = g.edata["bond_dist"][valid_three_body]
            l_g.ndata["pbc_offset"] = g.edata["pbc_offset"][valid_three_body]
        else:
            three_body_id = torch.unique(torch.concatenate(l_g.edges()))
            max_three_body_id = (
                torch.max(three_body_id) + 1 if three_body_id.numel() > 0 else 0
            )
            l_g.ndata["bond_vec"] = g.edata["bond_vec"][:max_three_body_id]
            l_g.ndata["bond_dist"] = g.edata["bond_dist"][:max_three_body_id]
            l_g.ndata["pbc_offset"] = g.edata["pbc_offset"][:max_three_body_id]
    l_g.apply_edges(compute_theta_and_phi)
    g.edata["rbf"] = expanded_dists
    three_body_basis = self.basis_expansion(l_g)
    three_body_cutoff = polynomial_cutoff(g.edata["bond_dist"], self.threebody_cutoff)
    node_feat, edge_feat, state_feat = self.embedding(
        node_types,
        g.edata["rbf"],
        state_attr,
    )
    for i in range(self.n_blocks):
        edge_feat = self.three_body_interactions[i](
            g,
            l_g,
            three_body_basis,
            three_body_cutoff,
            node_feat,
            edge_feat,
        )
        edge_feat, node_feat, state_feat = self.graph_layers[i](
            g,
            edge_feat,
            node_feat,
            state_feat,
        )
    g.ndata["node_feat"] = node_feat
    g.edata["edge_feat"] = edge_feat
    if self.is_intensive:
        node_vec = self.readout(g)
        vec = torch.hstack([node_vec, state_feat]) if self.include_states else node_vec  # type: ignore
        output = self.final_layer(vec)
        if self.task_type == "classification":
            output = self.sigmoid(output)
    else:
        g.ndata["atomic_properties"] = self.final_layer(g)
        output = dgl.readout_nodes(g, "atomic_properties", op="sum")

    return Embeddings(vec, node_vec)


M3GNet.m3gnet_forward = M3GNet.forward
M3GNet.forward = forward
