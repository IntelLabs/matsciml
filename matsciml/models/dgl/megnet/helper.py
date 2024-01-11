"""
Implementations of multi-layer perceptron (MLP) and other helper classes.

Code attributions to https://github.com/materialsvirtuallab/m3gnet-dgl/tree/main/megnet,
along with contributions and modifications from Marcel Nassar, Santiago Miret, and Kelvin Lee
"""
from __future__ import annotations

from typing import Callable, List, Optional

import torch
from dgl import DGLGraph, broadcast_edges, softmax_edges, sum_edges
from torch.nn import LSTM, Linear, Module, ModuleList


class MLP(Module):
    """
    An implementation of a multi-layer perception.
    """

    def __init__(
        self,
        dims: list[int],
        activation: Callable[[torch.Tensor], torch.Tensor] | None = None,
        activate_last: bool = False,
        bias_last: bool = True,
    ) -> None:
        """
        Bog standard MLP stack with activations in between.

        Parameters
        ----------
        dims : List[int]
            Dimensionality of each MLP layer
        activation : Optional[Callable[[torch.Tensor], torch.Tensor]], optional
            Activation functions to apply to each layer, by default None
        activate_last : bool, optional
            Whether to apply activations to the output layer, by default False
        bias_last : bool, optional
            Whether to include a bias term in the last layer, by default True
        """
        super().__init__()
        self._depth = len(dims) - 1
        self.layers = ModuleList()

        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            if i < self._depth - 1:
                self.layers.append(Linear(in_dim, out_dim, bias=True))

                if activation is not None:
                    self.layers.append(activation)
            else:
                self.layers.append(Linear(in_dim, out_dim, bias=bias_last))

                if activation is not None and activate_last:
                    self.layers.append(activation)

    def __repr__(self):
        dims = []

        for layer in self.layers:
            if isinstance(layer, Linear):
                dims.append(f"{layer.in_features} \u2192 {layer.out_features}")
            else:
                dims.append(layer.__class__.__name__)

        return f'MLP({", ".join(dims)})'

    @property
    def last_linear(self) -> Linear:
        """
        :return: The last linear layer.
        """
        for layer in reversed(self.layers):
            if isinstance(layer, Linear):
                return layer

    @property
    def depth(self) -> int:
        """Returns depth of MLP."""
        return self._depth

    @property
    def in_features(self) -> int:
        """Return input features of MLP"""
        return self.layers[0].in_features

    @property
    def out_features(self) -> int:
        """Returns output features of MLP."""
        for layer in reversed(self.layers):
            if isinstance(layer, Linear):
                return layer.out_features
        raise RuntimeError

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Applies all layers in turn.
        :param inputs: Input tensor
        :return: Output tensor
        """
        x = inputs

        for layer in self.layers:
            x = layer(x)

        return x


class EdgeSet2Set(Module):
    """
    Implementation of Set2Set.
    """

    def __init__(self, input_dim: int, n_iters: int, n_layers: int):
        """
        :param input_dim: The size of each input sample.
        :param n_iters: The number of iterations.
        :param n_layers: The number of recurrent layers.

        Parameters
        ----------
        input_dim : int
            The size of each input sample
        n_iters : int
            Number of iterations to operate on the Set2Set
        n_layers : int
            Number of recurrent (LSTM) layers
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = 2 * input_dim
        self.n_iters = n_iters
        self.n_layers = n_layers
        self.lstm = LSTM(self.output_dim, self.input_dim, n_layers)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reinitialize learnable parameters."""
        self.lstm.reset_parameters()

    def forward(self, graph: DGLGraph, feat: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass of the edge Set2Set.

        Parameters
        ----------
        graph : DGLGraph
            Input DGL graph structure
        feat : torch.Tensor
            Edge features to operate on

        Returns
        -------
        torch.Tensor
            Transformed features
        """
        with graph.local_scope():
            batch_size = graph.batch_size

            h = (
                feat.new_zeros((self.n_layers, batch_size, self.input_dim)),
                feat.new_zeros((self.n_layers, batch_size, self.input_dim)),
            )

            q_star = feat.new_zeros(batch_size, self.output_dim)

            for _ in range(self.n_iters):
                q, h = self.lstm(q_star.unsqueeze(0), h)
                q = q.view(batch_size, self.input_dim)
                e = (feat * broadcast_edges(graph, q)).sum(dim=-1, keepdim=True)
                graph.edata["e"] = e
                alpha = softmax_edges(graph, "e")
                graph.edata["r"] = feat * alpha
                readout = sum_edges(graph, "r")
                q_star = torch.cat([q, readout], dim=-1)

            return q_star
