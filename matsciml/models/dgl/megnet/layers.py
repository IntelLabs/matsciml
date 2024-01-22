"""
Implement graph convolution layers for MEGNet.

Code attributions to https://github.com/materialsvirtuallab/m3gnet-dgl/tree/main/megnet,
along with contributions and modifications from Marcel Nassar, Santiago Miret, and Kelvin Lee
"""
from __future__ import annotations

from typing import Dict, List, Optional

import dgl
import dgl.function as fn
import torch
from torch import nn
from torch.nn import Dropout, Identity, Module, Softplus

from matsciml.models.dgl.megnet import MLP


class MEGNetGraphConv(Module):
    """
    A MEGNet graph convolution layer in DGL.
    """

    def __init__(
        self,
        edge_func: nn.Module,
        node_func: nn.Module,
        attr_func: nn.Module,
    ) -> None:
        """
        Initialization method for the MEGNet graph convolution layer.

        Parameters
        ----------
        edge_func : nn.Module
            `nn.Module` function that operates on edge features.
        node_func : nn.Module
            `nn.Module` function that operates on node features.
        attr_func : nn.Module
            `nn.Module` function that operates on graph features.
        """
        super().__init__()
        self.edge_func = edge_func
        self.node_func = node_func
        self.attr_func = attr_func

    @classmethod
    def from_dims(
        cls,
        edge_dims: list[int],
        node_dims: list[int],
        attr_dims: list[int],
    ) -> MEGNetGraphConv:
        """
        Class method to instantiate a MEGNet graph convolution layer given
        a list of dimensionalities.

        Parameters
        ----------
        edge_dims, node_dims, attr_dims : list[int]
            Dimensionalities for each MLP function that transforms edge, node, and
            graph features.

        Returns
        -------
        MEGNetGraphConv
            Instance of a MEGNet graph convolution layer
        """
        # TODO(marcel): Softplus doesnt exactly match paper's SoftPlus2
        # TODO(marcel): Should we activate last?
        edge_update = MLP(edge_dims, Softplus(), activate_last=True)
        node_update = MLP(node_dims, Softplus(), activate_last=True)
        attr_update = MLP(attr_dims, Softplus(), activate_last=True)
        return cls(edge_update, node_update, attr_update)

    def _edge_udf(self, edges: dgl.udf.EdgeBatch) -> dict[str, torch.Tensor]:
        """
        Edge-based user defined function; will apply `edge_func` to edge features.

        Parameters
        ----------
        edges : dgl.udf.EdgeBatch
            Edge abstraction from DGL

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing the transformed edge features
        """
        vi = edges.src["v"]
        vj = edges.dst["v"]
        u = edges.src["u"]
        eij = edges.data.pop("e")
        inputs = torch.hstack([vi, vj, eij, u])
        mij = {"mij": self.edge_func(inputs)}
        return mij

    def edge_update_(self, graph: dgl.DGLGraph) -> torch.Tensor:
        """
        Update function for edges: applies the edge-focused MLP to get transformed
        edge features, and updates the corresponding key in the DGL graph.

        Parameters
        ----------
        graph : dgl.DGLGraph
            Input DGL graph

        Returns
        -------
        torch.Tensor
            Transformed edge features
        """
        graph.apply_edges(self._edge_udf)
        graph.edata["e"] = graph.edata.pop("mij")
        return graph.edata["e"]

    def node_update_(self, graph: dgl.DGLGraph) -> torch.Tensor:
        """
        Update function for nodes: applies the node-focused MLP to get transformed
        edge features, and updates the corresponding key in the DGL graph.

        Parameters
        ----------
        graph : dgl.DGLGraph
            Input DGL graph

        Returns
        -------
        torch.Tensor
            Transformed node features
        """
        graph.update_all(fn.copy_e("e", "e"), fn.mean("e", "ve"))
        ve = graph.ndata.pop("ve")
        v = graph.ndata.pop("v")
        u = graph.ndata.pop("u")
        inputs = torch.hstack([v, ve, u])
        graph.ndata["v"] = self.node_func(inputs)
        return graph.ndata["v"]

    def attr_update_(self, graph: dgl.DGLGraph, attrs: torch.Tensor) -> torch.Tensor:
        """
        Update function for graph attributes: applies the graph attribute
        focused MLP to get transformed graph features and returns them

        Parameters
        ----------
        graph : dgl.DGLGraph
            Input DGL graph

        attrs : torch.Tensor
            Input graph attributes

        Returns
        -------
        torch.Tensor
            Transformed graph features
        """
        u = attrs
        ue = dgl.readout_edges(graph, feat="e", op="mean")
        uv = dgl.readout_nodes(graph, feat="v", op="mean")
        inputs = torch.hstack([u, ue, uv])
        graph_attr = self.attr_func(inputs)
        return graph_attr

    def forward(
        self,
        graph: dgl.DGLGraph,
        edge_feat: torch.Tensor,
        node_feat: torch.Tensor,
        graph_attr: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for the MEGNet convolution operation, composing the
        order of the edge, node, and graph feature transformations.

        Parameters
        ----------
        graph : dgl.DGLGraph
            _description_
        edge_feat, node_feat, graph_attr : torch.Tensor
            Respective input feature tensors for each type

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Respective output tensors with transformed features.
        """
        with graph.local_scope():
            graph.edata["e"] = edge_feat
            graph.ndata["v"] = node_feat
            graph.ndata["u"] = dgl.broadcast_nodes(graph, graph_attr)

            edge_feat = self.edge_update_(graph)
            node_feat = self.node_update_(graph)
            graph_attr = self.attr_update_(graph, graph_attr)

        return edge_feat, node_feat, graph_attr


class MEGNetBlock(Module):
    """
    A MEGNet block comprising a sequence of update operations.
    """

    def __init__(
        self,
        dims: list[int],
        conv_hiddens: list[int],
        dropout: float | None = None,
        skip: bool = True,
    ) -> None:
        """
        MEGNet block init function

        Parameters
        ----------
        dims : list[int]
            List of integer dimensions to create MLP blocks
        conv_hiddens : list[int]
            Hidden dimension to use for the convolution layers
        dropout : Optional[float], optional
            Dropout proabibility, by default None which does not use
            dropout.
        skip : bool, optional
            Whether to use skip connections, by default True
        """
        super().__init__()

        self.has_dense = len(dims) > 1
        conv_dim = dims[-1]
        out_dim = conv_hiddens[-1]

        mlp_kwargs = {
            "dims": dims,
            "activation": Softplus(),
            "activate_last": True,
            "bias_last": True,
        }
        self.edge_func = MLP(**mlp_kwargs) if self.has_dense else Identity()
        self.node_func = MLP(**mlp_kwargs) if self.has_dense else Identity()
        self.attr_func = MLP(**mlp_kwargs) if self.has_dense else Identity()

        # compute input sizes
        edge_in = 2 * conv_dim + conv_dim + conv_dim  # 2*NDIM+EDIM+GDIM
        node_in = out_dim + conv_dim + conv_dim  # EDIM+NDIM+GDIM
        attr_in = out_dim + out_dim + conv_dim  # EDIM+NDIM+GDIM
        self.conv = MEGNetGraphConv.from_dims(
            edge_dims=[edge_in] + conv_hiddens,
            node_dims=[node_in] + conv_hiddens,
            attr_dims=[attr_in] + conv_hiddens,
        )

        self.dropout = Dropout(dropout) if dropout else None
        # TODO(marcel): should this be an 1D dropout
        self.skip = skip

    def forward(
        self,
        graph: dgl.DGLGraph,
        edge_feat: torch.Tensor,
        node_feat: torch.Tensor,
        graph_attr: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the MEGNet block, which transforms edge, node, and graph
        features based on the `MEGNetGraphConv` operations.

        Parameters
        ----------
        graph : dgl.DGLGraph
            Input DGL graph structure
        edge_feat, node_feat, graph_attr : torch.Tensor
            Respective feature tensors for each type

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Transformed feature tensors for each respective type of feature
        """

        inputs = (edge_feat, node_feat, graph_attr)
        edge_feat = self.edge_func(edge_feat)
        node_feat = self.node_func(node_feat)
        graph_attr = self.attr_func(graph_attr)

        edge_feat, node_feat, graph_attr = self.conv(
            graph,
            edge_feat,
            node_feat,
            graph_attr,
        )

        if self.dropout:
            edge_feat = self.dropout(edge_feat)
            node_feat = self.dropout(node_feat)
            graph_attr = self.dropout(graph_attr)

        if self.skip:
            edge_feat = edge_feat + inputs[0]
            node_feat = node_feat + inputs[1]
            graph_attr = graph_attr + inputs[2]

        return edge_feat, node_feat, graph_attr
