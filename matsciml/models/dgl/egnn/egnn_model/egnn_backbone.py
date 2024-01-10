# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: MIT License
from __future__ import annotations

from copy import deepcopy

import dgl
import torch
import torch.nn as nn
from dgl.nn.pytorch.glob import AvgPooling, SumPooling, WeightAndSum
from munch import Munch

from matsciml.models.dgl.egnn.egnn_model.nets import EGNN, MLP


# TODO (marcel): what needs to be done with pos when doing a node project
class EGNNBackbone(nn.Module):
    def __init__(
        self,
        egnn: nn.Module,
        node_projection: nn.Module,
        readout: nn.Module,
        prediction: nn.Module,
    ) -> nn.Module:
        super().__init__()
        self.egnn = egnn
        self.node_projection = node_projection
        self.readout = readout
        self.prediction = prediction

    def forward(
        self,
        graph: dgl.DGLGraph,
        node_feats: torch.Tensor,
        positions: torch.Tensor,
        edge_attributes: torch.Tensor,
    ) -> torch.Tensor:
        feats, _ = self.egnn(graph, node_feats, positions, edge_attributes)
        feats = self.node_projection(feats)
        feats = self.readout(graph, feats)
        feats = self.prediction(feats)

        return feats


def get_backbone(config_orig: Munch) -> nn.Module:
    config = deepcopy(config_orig)

    activations = {
        "gelu": nn.GELU(),
        "leaky_relu": nn.LeakyReLU(),
        "relu": nn.ReLU(),
        "silu": nn.SiLU(),
    }
    attention_norms = {"sigmoid": nn.Sigmoid(), "softmax": nn.Softmax(dim=-2)}

    config.embed.activation = activations[config.embed.activation]

    if config.embed.use_attention:
        config.embed.attention_norm = attention_norms[config.embed.attention_norm]

    egnn = EGNN(**config.embed)

    k_linears = config.embed.k_linears

    node_projection_hidden_dim = config.node_projection.hidden_dim
    node_projection_depth = config.node_projection.depth

    node_projection_dims = [config.embed.hidden_dim]
    node_projection_dims.extend(
        [node_projection_hidden_dim for _ in range(node_projection_depth)],
    )

    node_projection_activation = activations[config.node_projection.activation]

    node_projection = MLP(
        node_projection_dims,
        activation=node_projection_activation,
        activate_last=False,
        k_linears=k_linears,
    )

    prediction_hidden_dim = config.prediction.hidden_dim
    prediction_out_dim = config.prediction.out_dim
    prediction_depth = config.prediction.depth

    prediction_dims = [node_projection_dims[-1]]
    prediction_dims.extend([prediction_hidden_dim for _ in range(prediction_depth - 1)])
    prediction_dims.append(prediction_out_dim)

    prediction_activation = activations[config.prediction.activation]

    prediction = MLP(
        prediction_dims,
        activation=prediction_activation,
        activate_last=False,
        k_linears=k_linears,
    )

    if config.readout == "avg_pooling":
        readout = AvgPooling()
    elif config.readout == "sum_pooling":
        readout = SumPooling()
    elif config.readout == "weight_and_sum":
        readout = WeightAndSum(node_projection_dims[-1])

    backbone = EGNNBackbone(egnn, node_projection, readout, prediction)

    return backbone
