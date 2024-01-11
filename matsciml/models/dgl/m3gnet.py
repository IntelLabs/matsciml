from __future__ import annotations

from typing import Union

import dgl
import torch
from matgl.models import M3GNet


def forward(
    self,
    g: dgl.DGLGraph,
    state_attr: torch.Tensor | None = None,
    l_g: dgl.DGLGraph | None = None,
):
    g = g["graph"]
    return self.m3gnet_forward(g, state_attr, l_g)


M3GNet.m3gnet_forward = M3GNet.forward
M3GNet.forward = forward
