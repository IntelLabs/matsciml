# Copyright (C) 2022-2023 Intel Corporation
# SPDX-License-Identifier: MIT License
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Union

import dgl
import torch
import torch.nn as nn
from dgl.nn.pytorch.glob import AvgPooling, MaxPooling, SumPooling, WeightAndSum

from matsciml.common.types import BatchDict, DataDict, Embeddings
from matsciml.models.base import AbstractDGLModel
from matsciml.models.dgl.egnn.egnn_model import EGNN, MLP


class PLEGNNBackbone(AbstractDGLModel):
    def __init__(
        self,
        embed_in_dim: int,
        embed_hidden_dim: int,
        embed_out_dim: int,
        embed_depth: int,
        embed_feat_dims: list[int],
        embed_message_dims: list[int],
        embed_position_dims: list[int],
        embed_edge_attributes_dim: int,
        embed_activation: str,
        embed_residual: bool,
        embed_normalize: bool,
        embed_tanh: bool,
        embed_activate_last: bool,
        embed_k_linears: int,
        embed_use_attention: bool,
        embed_attention_norm: str,
        # readout
        readout: str,
        # node projection
        node_projection_depth: int,
        node_projection_hidden_dim: int,
        node_projection_activation: str,
        # prediction
        prediction_depth: int,
        prediction_hidden_dim: int,
        prediction_out_dim: int,
        prediction_activation: str,
        atom_embedding_dim: int | None = None,
        num_atom_embedding: int = 100,
        embedding_kwargs: dict[str, Any] = {},
        encoder_only: bool = True,
    ) -> None:
        super().__init__(
            embed_hidden_dim,
            num_atom_embedding,
            embedding_kwargs,
            encoder_only,
        )
        self.embed = EGNN(
            embed_in_dim,
            embed_hidden_dim,
            embed_out_dim,
            embed_depth,
            embed_feat_dims,
            embed_message_dims,
            embed_position_dims,
            embed_edge_attributes_dim,
            self._get_activation(embed_activation),
            embed_residual,
            embed_normalize,
            embed_tanh,
            activate_last=embed_activate_last,
            k_linears=embed_k_linears,
            use_attention=embed_use_attention,
            attention_norm=self._get_attention_norm(embed_attention_norm),
            num_atoms_embedding=num_atom_embedding,
        )
        self.embed.atom_embedding = self.atom_embedding

        self.encoder_only = encoder_only
        node_projection_dims = self._get_node_projection_dims(
            embed_hidden_dim,
            node_projection_depth,
            node_projection_hidden_dim,
        )
        self.node_projection = MLP(
            node_projection_dims,
            activation=self._get_activation(node_projection_activation),
            activate_last=False,
            k_linears=embed_k_linears,
        )

        self.readout = self._get_readout(readout, node_projection_dims[-1])

        if not encoder_only:
            prediction_dims = self._get_prediction_dims(
                node_projection_dims[-1],
                prediction_depth,
                prediction_hidden_dim,
                prediction_out_dim,
            )

            self.prediction = MLP(
                prediction_dims,
                activation=self._get_activation(prediction_activation),
                activate_last=False,
                k_linears=embed_k_linears,
            )

    @staticmethod
    def _get_activation(activation: str) -> Callable[[torch.Tensor], torch.Tensor]:
        activations = {
            "gelu": nn.GELU(),
            "leaky_relu": nn.LeakyReLU(),
            "relu": nn.ReLU(),
            "silu": nn.SiLU(),
        }

        return activations[activation]

    @staticmethod
    def _get_attention_norm(attention_norm: str):
        attention_norms = {
            "sigmoid": nn.Sigmoid(),
            "softmax": nn.Softmax(dim=-2),
        }

        return attention_norms[attention_norm]

    @staticmethod
    def _get_readout(readout: str, node_projection_last_dim: int):
        readouts = {
            "avg": AvgPooling(),
            "max": MaxPooling(),
            "sum": SumPooling(),
            "weight_and_sum": WeightAndSum(node_projection_last_dim),
        }

        return readouts[readout]

    @staticmethod
    def _get_node_projection_dims(
        embed_hidden_dim: int,
        node_projection_depth: int,
        node_projection_hidden_dim: int,
    ) -> list[int]:
        node_projection_dims = [embed_hidden_dim]
        node_projection_dims.extend(
            [node_projection_hidden_dim for _ in range(node_projection_depth)],
        )

        return node_projection_dims

    @staticmethod
    def _get_prediction_dims(
        node_projection_last_dim: int,
        prediction_depth: int,
        prediction_hidden_dim: int,
        prediction_out_dim: int,
    ):
        prediction_dims = [node_projection_last_dim]
        prediction_dims.extend(
            [prediction_hidden_dim for _ in range(prediction_depth - 1)],
        )
        prediction_dims.append(prediction_out_dim)

        return prediction_dims

    def read_batch(self, batch: BatchDict) -> DataDict:
        data = {}
        assert (
            "graph" in batch
        ), f"PLEGNN expects a DGLGraph in the 'graph' key of a batch."
        graph = batch.get("graph")
        atomic_numbers = graph.ndata["atomic_numbers"].long()
        pos = graph.ndata["pos"]
        data["graph"] = graph
        data["node_feats"] = atomic_numbers
        data["pos"] = pos
        # for now, EGNN assumes no edge features but can be setup
        data.setdefault("edge_feats", None)
        data.setdefault("graph_feats", None)
        return data

    def _forward(
        self,
        graph: dgl.DGLGraph,
        node_feats: torch.Tensor,
        pos: torch.Tensor,
        edge_feats: torch.Tensor | None = None,
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
        edge_feats : Optional[torch.Tensor], optional
            Tensor containing interatomic distances, by default None and unused.
        graph_feats : Optional[torch.Tensor], optional
            Graph-based properties, by default None and unused.

        Returns
        -------
        Embeddings
            Data structure containing node and graph level embeddings.
            Node embeddings correspond to after the node projection layer.
        """
        n_z, _ = self.embed(graph, node_feats, pos)
        n_z = self.node_projection(n_z)
        g_z = self.readout(graph, n_z)
        return Embeddings(g_z, n_z)
