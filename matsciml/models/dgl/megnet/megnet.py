# Copyright (C) 2022-3 Intel Corporation
# SPDX-License-Identifier: MIT License
"""
Implementation of MEGNet model.

Code attributions to https://github.com/materialsvirtuallab/m3gnet-dgl/tree/main/megnet,
along with contributions and modifications from Marcel Nassar, Santiago Miret, and Kelvin Lee
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import dgl
import torch
from dgl.nn import Set2Set
from torch import nn
from torch.nn import Dropout, Identity, Module, ModuleList, Softplus

from matsciml.common.types import BatchDict, DataDict, Embeddings
from matsciml.models.base import AbstractDGLModel
from matsciml.models.dgl.megnet import MLP, EdgeSet2Set, MEGNetBlock


class MEGNet(AbstractDGLModel):
    def __init__(
        self,
        edge_feat_dim: int,
        node_feat_dim: int,
        graph_feat_dim: int,
        num_blocks: int,
        hiddens: list[int],
        conv_hiddens: list[int],
        s2s_num_layers: int,
        s2s_num_iters: int,
        output_hiddens: list[int],
        is_classification: bool = True,
        node_embed: nn.Module | None = None,
        edge_embed: nn.Module | None = None,
        attr_embed: nn.Module | None = None,
        dropout: float | None = None,
        atom_embedding_dim: int | None = None,
        num_atom_embedding: int = 100,
        embedding_kwargs: dict[str, Any] = {},
        encoder_only: bool = True,
    ) -> None:
        r"""
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
        super().__init__(
            node_feat_dim,
            num_atom_embedding,
            embedding_kwargs,
            encoder_only,
        )
        self.edge_embed = edge_embed if edge_embed else Identity()
        # default behavior for node embeddings is to use a lookup table
        self.node_embed = node_embed if node_embed else self.atom_embedding
        self.attr_embed = attr_embed if attr_embed else Identity()

        self.edge_encoder = MLP(
            [edge_feat_dim] + hiddens,
            Softplus(),
            activate_last=True,
        )
        # in the event we're using an embedding table, skip the input dim because
        # we're using the hidden dimensionality
        if isinstance(self.node_embed, nn.Embedding):
            node_encoder = MLP(
                [node_feat_dim + 3] + hiddens,
                Softplus(),
                activate_last=True,
            )
        else:
            node_encoder = MLP(
                [node_feat_dim] + hiddens,
                Softplus(),
                activate_last=True,
            )
        self.node_encoder = node_encoder
        self.attr_encoder = MLP(
            [graph_feat_dim] + hiddens,
            Softplus(),
            activate_last=True,
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

        self.encoder_only = encoder_only
        if not encoder_only:
            self.output_proj = MLP(
                # S2S cats q_star to output producing double the dim
                dims=[2 * 2 * block_out_dim + block_out_dim] + output_hiddens + [1],
                activation=Softplus(),
                activate_last=False,
            )

        self.dropout = Dropout(dropout) if dropout else None
        # TODO(marcel): should this be an 1D dropout

        self.is_classification = is_classification

    def read_batch(self, batch: BatchDict) -> DataDict:
        r"""
        Extracts data needed by MEGNet from the batch and graph
        structures.

        In particular, we pack node features as positions + atom embeddings,
        and looks for graph level variables alongside edge data 'r' and 'mu'.

        Parameters
        ----------
        batch : BatchDict
            Batch of data to be processed

        Returns
        -------
        DataDict
            Input data for MEGNet as a dictionary.
        """
        data = super().read_batch(batch)
        graph = data.get("graph")
        # stack atom embeddings from table and positions together
        node_feats = self.join_position_embeddings(
            graph.ndata["pos"],
            data["node_feats"],
        )
        data["node_feats"] = node_feats
        assert (
            "graph_variables" in batch
        ), f"MEGNet expects graph level features. Please include 'GraphVariablesTransform in your data pipeline."
        data["graph_feats"] = batch.get("graph_variables")
        assert (
            "r" in graph.edata
        ), f"MEGNet expects interatomic distances in edge data. Please include 'DistancesTransform' in your data pipeline."
        assert (
            "mu" in graph.edata
        ), f"MEGNet expects reduced masses in edge data. Please include 'DistancesTransform' in your data pipeline."
        edge_feats = torch.hstack([graph.edata["r"], graph.edata["mu"].unsqueeze(-1)])
        data["edge_feats"] = edge_feats
        return data

    def _forward(
        self,
        graph: dgl.DGLGraph,
        node_feats: torch.Tensor,
        edge_feats: torch.Tensor,
        graph_feats: torch.Tensor,
        pos: torch.Tensor | None = None,
        **kwargs,
    ) -> Embeddings:
        r"""
        Implement the forward method, which computes the energy of
        a molecular graph.

        Parameters
        ----------
        graph : dgl.DGLGraph
            A single or batch of molecular graphs

        Parameters
        ----------
        graph : dgl.DGLGraph
            Instance of a DGL graph data structure
        node_feats : torch.Tensor
            Atomic embeddings obtained from nn.Embedding
        edge_feats : torch.Tensor
            Tensor containing interatomic distances
        graph_feats : torch.Tensor
            Graph-based properties
        pos : Optional[torch.Tensor], optional
            XYZ coordinates of each atom, by default None and unused.

        Returns
        -------
        Embeddings
            Data structure containing graph and node level embeddings.
        """
        edge_feats = self.edge_encoder(self.edge_embed(edge_feats))
        node_feats = self.node_encoder(node_feats)
        graph_feats = self.attr_encoder(self.attr_embed(graph_feats))

        for block in self.blocks:
            output = block(graph, edge_feats, node_feats, graph_feats)
            edge_feats, node_feats, graph_feats = output

        node_vec = self.node_s2s(graph, node_feats)
        edge_vec = self.edge_s2s(graph, edge_feats)

        vec = torch.hstack([node_vec, edge_vec, graph_feats])

        if self.dropout:
            vec = self.dropout(vec)  # pylint: disable=E1102
        embeddings = Embeddings(vec, node_vec)
        return embeddings
