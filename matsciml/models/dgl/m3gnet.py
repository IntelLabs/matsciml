from __future__ import annotations


import dgl
import torch
from matgl.models import M3GNet

from matsciml.common.types import BatchDict

from matsciml.common.types import Embeddings

"""
M3GNet is integrated from matgl: https://github.com/materialsvirtuallab/matgl
The forward pass needs to be overwritten to accommodate matsciml's expected inputs,
and to construct the Embedding's output object.
"""


def forward(
    self,
    batch: BatchDict,
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
    graph = batch["graph"]
    outputs = self.m3gnet_forward(graph, state_attr, l_g, return_all_layer_output=True)
    # gc_3 is essentially the last graph layer before the readout
    return Embeddings(outputs["readout"], outputs["gc_3"]["node_feat"])


M3GNet.m3gnet_forward = M3GNet.forward
M3GNet.forward = forward
