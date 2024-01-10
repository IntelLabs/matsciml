# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: MIT License
from __future__ import annotations

from importlib.util import find_spec
from typing import Any, Dict, List, Optional, Type, Union

import dgl
import torch
from dgl.nn.pytorch import glob
from dgllife.model import MPNNGNN
from torch import nn

from matsciml.common.types import BatchDict, DataDict, Embeddings
from matsciml.models.base import AbstractDGLModel


class MPNN(AbstractDGLModel):
    def __init__(
        self,
        atom_embedding_dim: int,
        edge_in_dim: int = 1,
        node_out_dim: int = 64,
        edge_out_dim: int = 128,
        num_step_message_passing: int = 3,
        num_atom_embedding: int = 100,
        embedding_kwargs: dict[str, Any] = {},
        readout: type[nn.Module] | str | nn.Module = glob.AvgPooling,
        readout_kwargs: dict[str, Any] | None = None,
        encoder_only: bool = True,
    ) -> None:
        r"""
        _summary_

        Parameters
        ----------
        atom_embedding_dim : int
            Dimensionality of atom vector embeddings
        edge_in_dim : int, optional
            Dimensionality of edge features, by default 1, corresponding
            with interatomic distances
        node_out_dim : int, optional
            Output dimensionality of node features, by default 64
        edge_out_dim : int, optional
            Output dimensionality of edge features, by default 128
        num_step_message_passing : int, optional
            Number of message passing steps, by default 3
        num_atom_embedding : int, optional
            Number of elements in the embedding table, by default 100
        embedding_kwargs : Dict[str, Any], optional
            Kwargs to be passed into the embedding table, by default ...
        readout : Union[Type[nn.Module], str, nn.Module], optional
            Aggregation function for node to graph embedding, by default glob.AvgPooling
        readout_kwargs : Optional[Dict[str, Any]], optional
            Kwargs to be passed into readout object instantiation, by default None
        encoder_only : bool, optional
            If True, bypasses the output projection layer, by default True
        """
        super().__init__(
            atom_embedding_dim,
            num_atom_embedding,
            embedding_kwargs,
            encoder_only,
        )
        self.model = MPNNGNN(
            atom_embedding_dim + 3,
            edge_in_dim,
            node_out_dim,
            edge_out_dim,
            num_step_message_passing,
        )
        if isinstance(readout, (str, type)):
            # if str, assume it's the name of a class
            if isinstance(readout, str):
                readout_cls = find_spec(readout, "dgl.nn.pytorch.glob")
                if readout_cls is None:
                    raise ImportError(
                        f"Class name passed to `readout`, but not found in `dgl.nn.pytorch.glob`.",
                    )
            else:
                # assume it's generic type
                readout_cls = readout
            if readout_kwargs is None:
                readout_kwargs = {}
            readout = readout_cls(**readout_kwargs)
        self.readout = readout
        self.encoder_only = encoder_only
        self.output = nn.Linear(node_out_dim, 1)

    def read_batch(self, batch: BatchDict) -> DataDict:
        r"""
        Adds an expectation for interatomic distances in the graph edge data,
        needed by the MPNN model.

        Parameters
        ----------
        batch : BatchDict
            Batch of data to be processed

        Returns
        -------
        DataDict
            Input data to be passed into MPNN
        """
        data = super().read_batch(batch)
        graph = data["graph"]
        assert (
            "r" in graph.edata
        ), "Expected 'r' key in graph edge data. Please include 'DistancesTransform' in data definition."
        data["edge_feats"] = graph.edata["r"]
        return data

    def _forward(
        self,
        graph: dgl.DGLGraph,
        node_feats: torch.Tensor,
        pos: torch.Tensor,
        edge_feats: torch.Tensor,
        graph_feats: torch.Tensor | None = None,
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
        pos : torch.Tensor
            XYZ coordinates of each atom
        edge_feats : torch.Tensor
            Tensor containing interatomic distances
        graph_feats : Optional[torch.Tensor], optional
            Graph-based properties, by default None and unused.

        Returns
        -------
        Embeddings
            Data structure with graph and node level embeddings packed.
            Node embeddings are from the last message passing layer.
        """
        node_feats = self.join_position_embeddings(pos, node_feats)
        n_z = self.model(graph, node_feats, edge_feats)
        g_z = self.readout(graph, n_z)
        embeddings = Embeddings(g_z, n_z)
        return embeddings
