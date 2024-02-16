from __future__ import annotations

import dgl
import torch

from matgl.graph.compute import (
    compute_pair_vector_and_distance,
)
from matgl.models import TensorNet as MGLTensorNet
from matgl.utils.maths import decompose_tensor, tensor_norm

from matsciml.common.types import Embeddings
from matsciml.common.registry import registry
from matsciml.models.base import AbstractDGLModel

__all__ = ["TensorNet"]


@registry.register_model("TensorNet")
class TensorNet(AbstractDGLModel, MGLTensorNet):
    def __init__(self, element_types: list[str], *args, **kwargs):
        super().__init__(atom_embedding_dim=len(element_types))
        self.tensor_embedding.emb = torch.nn.Embedding(len(element_types), self.units)
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
        outputs = self.tensornet_forward(graph, **kwargs)
        return Embeddings(outputs[0], outputs[1])

    def tensornet_forward(
        self, g: dgl.DGLGraph, state_attr: torch.Tensor | None = None, **kwargs
    ):
        """

        Args:
            g : DGLGraph for a batch of graphs.
            state_attr: State attrs for a batch of graphs.
            **kwargs: For future flexibility. Not used at the moment.

        Returns:
            output: output: Output property for a batch of graphs
        """
        # Obtain graph, with distances and relative position vectors
        g.edata["pbc_offshift"] = g.edata["offsets"]
        g.ndata["node_type"] = g.ndata["atomic_numbers"].long()
        bond_vec, bond_dist = compute_pair_vector_and_distance(g)
        g.edata["bond_vec"] = bond_vec.to(g.device)
        g.edata["bond_dist"] = bond_dist.to(g.device)

        # This asserts convinces TorchScript that edge_vec is a Tensor and not an Optional[Tensor]

        # Expand distances with radial basis functions
        edge_attr = self.bond_expansion(g.edata["bond_dist"])
        g.edata["edge_attr"] = edge_attr
        # Embedding from edge-wise tensors to node-wise tensors
        X, edge_feat, state_feat = self.tensor_embedding(g, state_attr)
        # Interaction layers
        for layer in self.layers:
            X = layer(g, X)
        scalars, skew_metrices, traceless_tensors = decompose_tensor(X)

        x = torch.cat(
            (
                tensor_norm(scalars),
                tensor_norm(skew_metrices),
                tensor_norm(traceless_tensors),
            ),
            dim=-1,
        )
        x = self.out_norm(x)

        g.ndata["node_feat"] = x
        if self.is_intensive:
            node_vec = self.readout(g)
            vec = node_vec  # type: ignore
            output = self.final_layer(vec)
            if self.task_type == "classification":
                output = self.sigmoid(output)
            matsciml_output = (vec, node_vec)
            return matsciml_output
        g.ndata["atomic_properties"] = self.final_layer(g)
        output = dgl.readout_nodes(g, "atomic_properties", op="sum")
        matsciml_output = (output, g.ndata["atomic_properties"])
        return matsciml_output
