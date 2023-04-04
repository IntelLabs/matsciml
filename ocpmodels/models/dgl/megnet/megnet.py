"""
Implementation of MEGNet model.

Code attributions to https://github.com/materialsvirtuallab/m3gnet-dgl/tree/main/megnet,
along with contributions and modifications from Marcel Nassar, Santiago Miret, and Kelvin Lee
"""
from typing import Optional, List

import dgl
import torch
from torch import nn
from dgl.nn import Set2Set
from torch.nn import Dropout, Identity, Module, ModuleList, Softplus

from ocpmodels.models.dgl.megnet import MLP, MEGNetBlock, EdgeSet2Set
from ocpmodels.models import AbstractEnergyModel


class MEGNet(AbstractEnergyModel):
    """
    DGL implementation of MEGNet.
    """

    def __init__(
        self,
        edge_feat_dim: int,
        node_feat_dim: int,
        graph_feat_dim: int,
        num_blocks: int,
        hiddens: List[int],
        conv_hiddens: List[int],
        s2s_num_layers: int,
        s2s_num_iters: int,
        output_hiddens: List[int],
        is_classification: bool = True,
        node_embed: Optional[nn.Module] = None,
        edge_embed: Optional[nn.Module] = None,
        attr_embed: Optional[nn.Module] = None,
        dropout: Optional[float] = None,
        num_atom_embedding: int = 100,
    ) -> None:
        """
        Init method for MEGNet. Also supports learnable embeddings for each
        atom, as specified by `num_atom_embedding` for the number of types of
        atoms. The embedding dimensionality is given by the first element of
        the `hiddens` arg.

        Parameters
        ----------
        in_dim : int
            Input dimensionality, which is used to create the encoder layers.
        num_blocks : int
            Number of MEGNet convolution blocks to use
        hiddens : List[int]
            Hidden dimensionality of encoding MLP layers, follows `in_dim`
        conv_hiddens : List[int]
            Hidden dimensionality of the convolution layers
        s2s_num_layers : int
            Number of Set2Set layers
        s2s_num_iters : int
            Number of iterations for Set2Set operations
        output_hiddens : List[int]
            Output layer hidden dimensionality in the projection layer
        is_classification : bool, optional
            Whether to apply sigmoid to the output tensor, by default True
        node_embed, edge_embed, attr_embed : Optional[nn.Module], optional
            Embedding functions for each type of feature, by default None and
            simply uses an `Identity` function
        dropout : Optional[float], optional
            Dropout probability for the convolution layers, by default None
            which does not use dropout.
        num_atom_embedding : int, optional
            Number of embeddings to use for the atom node embedding table, by
            default is 100.
        """
        super().__init__()

        self.edge_embed = edge_embed if edge_embed else Identity()
        # default behavior for node embeddings is to use a lookup table
        self.node_embed = (
            node_embed if node_embed else nn.Embedding(num_atom_embedding, hiddens[0])
        )
        self.attr_embed = attr_embed if attr_embed else Identity()

        self.edge_encoder = MLP(
            [edge_feat_dim] + hiddens, Softplus(), activate_last=True
        )
        # in the event we're using an embedding table, skip the input dim because
        # we're using the hidden dimensionality
        if isinstance(self.node_embed, nn.Embedding):
            node_encoder = MLP([hiddens[0] + 3] + hiddens, Softplus(), activate_last=True)
        else:
            node_encoder = MLP(
                [node_feat_dim] + hiddens, Softplus(), activate_last=True
            )
        self.node_encoder = node_encoder
        self.attr_encoder = MLP(
            [graph_feat_dim] + hiddens, Softplus(), activate_last=True
        )

        blocks_in_dim = hiddens[-1]
        block_out_dim = conv_hiddens[-1]
        block_args = dict(conv_hiddens=conv_hiddens, dropout=dropout, skip=True)
        blocks = []

        # first block
        blocks.append(MEGNetBlock(dims=[blocks_in_dim], **block_args))  # type: ignore
        # other blocks
        for _ in range(num_blocks - 1):
            blocks.append(MEGNetBlock(dims=[block_out_dim] + hiddens, **block_args))  # type: ignore
        self.blocks = ModuleList(blocks)

        s2s_kwargs = dict(n_iters=s2s_num_iters, n_layers=s2s_num_layers)
        self.edge_s2s = EdgeSet2Set(block_out_dim, **s2s_kwargs)
        self.node_s2s = Set2Set(block_out_dim, **s2s_kwargs)

        self.output_proj = MLP(
            # S2S cats q_star to output producing double the dim
            dims=[2 * 2 * block_out_dim + block_out_dim] + output_hiddens + [1],
            activation=Softplus(),
            activate_last=False,
        )

        self.dropout = Dropout(dropout) if dropout else None
        # TODO(marcel): should this be an 1D dropout

        self.is_classification = is_classification

    def forward(
        self,
        graph: dgl.DGLGraph,
        edge_feat: torch.Tensor,
        node_labels: torch.Tensor,
        node_pos: torch.Tensor,
        graph_attr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of MEGNet, taking in an input DGL graph and
        transforming the input features with encoding layers first,
        followed by blocks of graph convolution and projection.

        Parameters
        ----------
        graph : dgl.DGLGraph
            _description_
        edge_feat, node_feat, graph_attr : torch.Tensor
            Respective feature tensors for each type of representation

        Returns
        -------
        torch.Tensor
            Output tensor, typically is the energy.
        """
        # in the event we're using an embedding table, make sure we're
        # casting the node features correctly
        atom_embeddings = self.node_embed(node_labels)
        node_feat = torch.hstack([node_pos, atom_embeddings])

        edge_feat = self.edge_encoder(self.edge_embed(edge_feat))
        node_feat = self.node_encoder(node_feat)
        graph_attr = self.attr_encoder(self.attr_embed(graph_attr))

        for block in self.blocks:
            output = block(graph, edge_feat, node_feat, graph_attr)
            edge_feat, node_feat, graph_attr = output

        node_vec = self.node_s2s(graph, node_feat)
        edge_vec = self.edge_s2s(graph, edge_feat)

        vec = torch.hstack([node_vec, edge_vec, graph_attr])

        if self.dropout:
            vec = self.dropout(vec)  # pylint: disable=E1102

        output = self.output_proj(vec)
        if self.is_classification:
            output = torch.sigmoid(output)

        return output
