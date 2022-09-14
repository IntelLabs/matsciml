# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: MIT License

from typing import Callable, List

import dgl
import torch
import torch.nn as nn
from dgl.nn.pytorch.glob import AvgPooling, MaxPooling, SumPooling, WeightAndSum

from ocpmodels.models import AbstractEnergyModel
from ocpmodels.models.dgl.egnn.egnn_model import EGNN, MLP


class PLEGNNBackbone(AbstractEnergyModel):
    def __init__(
        self,
        # embed
        embed_in_dim: int,
        embed_hidden_dim: int,
        embed_out_dim: int,
        embed_depth: int,
        embed_feat_dims: List[int],
        embed_message_dims: List[int],
        embed_position_dims: List[int],
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
    ) -> None:
        super().__init__()
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
        )

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
    ) -> List[int]:
        node_projection_dims = [embed_hidden_dim]
        node_projection_dims.extend(
            [node_projection_hidden_dim for _ in range(node_projection_depth)]
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
            [prediction_hidden_dim for _ in range(prediction_depth - 1)]
        )
        prediction_dims.append(prediction_out_dim)

        return prediction_dims

    def forward(self, graph: dgl.DGLGraph) -> torch.Tensor:
        inputs = graph.ndata["atomic_numbers"].unsqueeze(-1)
        pos = graph.ndata["pos"]

        x, _ = self.embed(graph, inputs, pos)
        x = self.node_projection(x)
        x = self.readout(graph, x)
        x = self.prediction(x)

        return x
