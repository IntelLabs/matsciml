# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: MIT License
from __future__ import annotations

import math
from typing import Dict, Tuple

import dgl
import dgl.function as fn
import torch
import torch.nn as nn


class KLinears(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        k: int,
        bias: bool = True,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ) -> nn.Module:
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(
                (k, in_features, out_features),
                device=device,
                dtype=dtype,
            ),
        )

        if bias:
            self.bias = nn.Parameter(
                torch.empty(
                    (k, 1, out_features),
                    device=device,
                    dtype=dtype,
                ),
            )
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        assert self.weight[0].dim() == 2

        for weight in self.weight:
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5))

        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[0])
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0

            nn.init.uniform_(self.bias, a=-bound, b=bound)

    @property
    def k(self) -> int:
        return self.weight.shape[0]

    @property
    def shape(self) -> tuple[int]:
        return tuple(self.weight.shape) + (self.bias is not None,)

    @property
    def in_features(self) -> tuple[int]:
        return self.shape[:2][0], self.shape[:2][1]

    @property
    def out_features(self) -> tuple[int]:
        return self.shape[0], self.shape[2]

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.dim() < 3:
            x = inputs.expand(self.k, -1, -1)
        elif inputs.dim() == 3:
            x = inputs.transpose(0, 1)
        else:
            raise TypeError("Invalid inputs tensor.")

        x = torch.bmm(x, self.weight)

        if self.bias is not None:
            x += self.bias

        x = x.transpose(0, 1)

        return x


class EquiCoordGraphConv(nn.Module):
    def __init__(
        self,
        edge_func: nn.Module,
        position_func: nn.Module,
        feat_func: nn.Module,
        attention_func: nn.Module = None,
        residual: bool = True,
        normalize: bool = True,
        tanh: bool = True,
    ) -> nn.Module:
        super().__init__()
        self.edge_func = edge_func
        self.position_func = position_func
        self.feat_func = feat_func
        self.attention_func = attention_func
        self.residual = residual
        self.normalize = normalize
        self.tanh = tanh

        self._verify()
        self.reset_parameters()

    def _verify(self):
        position_last = self.position_func.last_linear
        position_last_out_dim = position_last.out_features

        if isinstance(position_last, KLinears):
            position_last_out_dim = position_last_out_dim[1]

        exception_message = []

        if position_last_out_dim != 1:
            exception_message.append(
                f"out dim ({position_last_out_dim}) " f"doesn't equal (1)",
            )

        if position_last.bias is not None:
            exception_message.append("and last layer include bias")

        if exception_message:
            exception_message.insert(0, "Invalid coordinate update network:")

            raise TypeError(" ".join(exception_message))

    def reset_parameters(self):
        position_last = self.position_func.last_linear

        nn.init.xavier_uniform_(position_last.weight, gain=0.001)

    def _columnize(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs if inputs.dim() > 1 else inputs.unsqueeze(dim=1)

        return x

    def _edge_message(self, edges: dgl.udf.EdgeBatch) -> dict[str, torch.Tensor]:
        hi = self._columnize(edges.src["h"])
        hj = self._columnize(edges.dst["h"])

        try:
            aij = self._columnize(edges.data["a"])
        except KeyError as e:
            aij = None

        distance = edges.data.pop("r") ** 2
        inputs = [x for x in [hi, hj, distance, aij] if torch.is_tensor(x)]
        inputs = torch.cat(inputs, dim=-1)

        try:
            mij = self.edge_func(inputs)
        except RuntimeError as e:
            raise RuntimeError(
                f"Inputs last dim ({inputs.shape[-1]}) "
                f"doesn't match edge_func in dim "
                f"({self.edge_func.in_features})",
            ) from e

        if self.attention_func is not None:
            attention_score = self.attention_func(mij)

            mij = attention_score * mij

        mij = {"mij": mij}

        return mij

    def _feat_update(self, graph: dgl.DGLGraph) -> torch.Tensor:
        graph.update_all(fn.copy_e("mij", "mij"), fn.sum("mij", "mi"))

        inputs = torch.cat([graph.ndata["h"], graph.ndata.pop("mi")], dim=-1)
        x = self.feat_func(inputs)

        if self.residual:
            x += graph.ndata["h"]

        return x

    def _coordinate_update(self, graph: dgl.DGLGraph) -> torch.Tensor:
        weights = self.position_func(graph.edata.pop("mij"))

        if self.tanh:
            weights = torch.tanh(weights)

        graph.edata["(xi-xj)*phi(mij)"] = graph.edata["xi-xj"] * weights
        graph.update_all(
            fn.copy_e("(xi-xj)*phi(mij)", "m"),
            fn.sum("m", "update"),
        )

        x = graph.ndata["x"] + graph.ndata.pop("update")

        return x

    def forward(
        self,
        graph: dgl.DGLGraph,
        node_feats: torch.Tensor,
        positions: torch.Tensor,
        edge_attributes: torch.Tensor = None,
    ) -> tuple[torch.Tensor]:
        with graph.local_scope():
            graph.ndata["h"] = self._columnize(node_feats)
            graph.ndata["x"] = positions

            if edge_attributes is not None:
                graph.edata["a"] = edge_attributes

            graph.apply_edges(fn.u_sub_v("x", "x", "xi-xj"))
            graph.edata["r"] = torch.linalg.norm(
                graph.edata["xi-xj"],
                dim=-1,
                keepdim=True,
            )

            if self.normalize and graph.num_nodes() > 1:
                graph.edata["xi-xj"] = graph.edata["xi-xj"] / (graph.edata["r"] + 10e-2)

            graph.apply_edges(self._edge_message)

            feats = self._feat_update(graph)
            pos = self._coordinate_update(graph)

        return feats, pos
