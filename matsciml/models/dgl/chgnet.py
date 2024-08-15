from __future__ import annotations

import dgl
from dgl import readout_edges, readout_nodes
import torch

from matgl.graph.compute import (
    compute_pair_vector_and_distance,
    compute_theta,
    create_line_graph,
    ensure_line_graph_compatibility,
)
from matgl.utils.cutoff import polynomial_cutoff
from matgl.models import CHGNet as MGLCHGNet

from matsciml.common.types import Embeddings
from matsciml.common.registry import registry
from matsciml.models.base import AbstractDGLModel

__all__ = ["CHGNet"]


@registry.register_model("CHGNet")
class CHGNet(AbstractDGLModel, MGLCHGNet):
    def __init__(self, element_types: list[str], *args, **kwargs):
        super().__init__(atom_embedding_dim=len(element_types))
        self.atom_embedding = MGLCHGNet(element_types).atom_embedding
        self.element_types = element_types

    def _forward(
        self,
        graph: dgl.DGLGraph,
        node_feats: torch.Tensor,
        edge_feats: torch.Tensor,
        graph_feats: torch.Tensor,
        pos: torch.Tensor | None = None,
        **kwargs,
    ) -> Embeddings:
        outputs = self.chgnet_forward(graph, **kwargs)
        return Embeddings(outputs[0], outputs[1])

    def chgnet_forward(
        self,
        g: dgl.DGLGraph,
        state_attr: torch.Tensor | None = None,
        l_g: dgl.DGLGraph | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass of the model.

        Args:
            g (dgl.DGLGraph): Input g.
            state_attr (torch.Tensor, optional): State features. Defaults to None.
            l_g (dgl.DGLGraph, optional): Line graph. Defaults to None and is computed internally.

        Returns:
            torch.Tensor: Model output.
        """
        # compute bond vectors and distances and add to g, needs to be computed here to register gradients
        bond_vec, bond_dist = compute_pair_vector_and_distance(g)
        g.edata["bond_vec"] = bond_vec.to(g.device)
        g.edata["bond_dist"] = bond_dist.to(g.device)
        bond_expansion = self.bond_expansion(bond_dist)
        smooth_cutoff = polynomial_cutoff(
            bond_expansion, self.cutoff, self.cutoff_exponent
        )
        g.edata["bond_expansion"] = smooth_cutoff * bond_expansion

        # create bond graph (line graoh) with necessary node and edge data
        if l_g is None:
            bond_graph = create_line_graph(g, self.three_body_cutoff, directed=True)
        else:
            # need to ensure the line graph matches the graph
            bond_graph = ensure_line_graph_compatibility(
                g, l_g, self.three_body_cutoff, directed=True
            )

        bond_graph.ndata["bond_index"] = bond_graph.ndata["edge_ids"]
        threebody_bond_expansion = self.threebody_bond_expansion(
            bond_graph.ndata["bond_dist"]
        )
        smooth_cutoff = polynomial_cutoff(
            threebody_bond_expansion, self.three_body_cutoff, self.cutoff_exponent
        )
        bond_graph.ndata["bond_expansion"] = smooth_cutoff * threebody_bond_expansion
        # the center atom is the dst atom of the src bond or the reverse (the src atom of the dst bond)
        # need to use "bond_index" just to be safe always
        bond_indices = bond_graph.ndata["bond_index"][bond_graph.edges()[0]]
        bond_graph.edata["center_atom_index"] = g.edges()[1][bond_indices]
        bond_graph.apply_edges(compute_theta)
        bond_graph.edata["angle_expansion"] = self.angle_expansion(
            bond_graph.edata["theta"]
        )

        # compute state, atom, bond and angle embeddings
        atom_features = self.atom_embedding(g.ndata["node_type"])
        bond_features = self.bond_embedding(g.edata["bond_expansion"])
        angle_features = self.angle_embedding(bond_graph.edata["angle_expansion"])
        if self.state_embedding is not None and state_attr is not None:
            state_attr = self.state_embedding(state_attr)
        else:
            state_attr = None

        # shared message weights
        atom_bond_weights = (
            self.atom_bond_weights(g.edata["bond_expansion"])
            if self.atom_bond_weights is not None
            else None
        )
        bond_bond_weights = (
            self.bond_bond_weights(g.edata["bond_expansion"])
            if self.bond_bond_weights is not None
            else None
        )
        threebody_bond_weights = (
            self.threebody_bond_weights(bond_graph.ndata["bond_expansion"])
            if self.threebody_bond_weights is not None
            else None
        )

        # message passing layers
        for i in range(self.n_blocks - 1):
            atom_features, bond_features, state_attr = self.atom_graph_layers[i](
                g,
                atom_features,
                bond_features,
                state_attr,
                atom_bond_weights,
                bond_bond_weights,
            )
            bond_features, angle_features = self.bond_graph_layers[i](
                bond_graph,
                atom_features,
                bond_features,
                angle_features,
                threebody_bond_weights,
            )

        # site wise target readout
        g.ndata["magmom"] = self.sitewise_readout(atom_features)

        # last atom graph message passing layer
        atom_features, bond_features, state_attr = self.atom_graph_layers[-1](
            g,
            atom_features,
            bond_features,
            state_attr,
            atom_bond_weights,
            bond_bond_weights,
        )

        # really only needed if using the readout modules in _readout.py
        # g.ndata["node_feat"] = atom_features
        # g.edata["edge_feat"] = bond_features
        # bond_graph.edata["angle_features"] = angle_features

        # readout
        if self.readout_field == "atom_feat":
            g.ndata["atom_feat"] = self.final_layer(atom_features)
            structure_properties = readout_nodes(
                g, "atom_feat", op=self.readout_operation
            )
        elif self.readout_field == "bond_feat":
            g.edata["bond_feat"] = self.final_layer(bond_features)
            structure_properties = readout_edges(
                g, "bond_feat", op=self.readout_operation
            )
        else:  # self.readout_field == "angle_feat":
            bond_graph.edata["angle_feat"] = self.final_layer(angle_features)
            structure_properties = readout_edges(
                bond_graph, "angle_feat", op=self.readout_operation
            )

        structure_properties = torch.squeeze(structure_properties)
        matsciml_output = (structure_properties, atom_features)
        return matsciml_output
