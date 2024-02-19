from __future__ import annotations


import dgl
import torch

from matsciml.common.types import Embeddings


from matgl.models import M3GNet as matgl_m3gnet
from matsciml.common.registry import registry
from matsciml.models.base import AbstractDGLModel

"""
M3GNet is integrated from matgl: https://github.com/materialsvirtuallab/matgl
The forward pass needs to be overwritten to accommodate matsciml's expected inputs,
and to construct the Embedding's output object.
"""


@registry.register_model("M3GNet")
class M3GNet(AbstractDGLModel):
    def __init__(
        self, element_types: list[str], return_all_layer_output: bool, *args, **kwargs
    ):
        super().__init__(atom_embedding_dim=len(element_types))
        self.elemenet_types = element_types
        self.all_embeddings = return_all_layer_output
        self.model = matgl_m3gnet(element_types, *args, **kwargs)
        self.atomic_embedding = self.model.embedding

    def _forward(
        self,
        graph: dgl.DGLGraph,
        node_feats: torch.Tensor,
        edge_feats: torch.Tensor,
        graph_feats: torch.Tensor,
        pos: torch.Tensor | None = None,
        **kwargs,
    ) -> Embeddings:
        outputs = self.model(
            graph, return_all_layer_output=self.all_embeddings, **kwargs
        )
        # gc_{self.model.n_blocks} is essentially the last graph layer before the readout
        last_layer = f"gc_{self.model.n_blocks}"
        return Embeddings(outputs["readout"], outputs[last_layer]["node_feat"])
