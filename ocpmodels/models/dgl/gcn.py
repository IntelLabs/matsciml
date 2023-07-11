# Copyright (C) 2022-3 Intel Corporation
# SPDX-License-Identifier: MIT License

"""
This module shows the pattern for defining new GNNs to be plugged into
either S2EF/IS2RE tasks.

The general abstraction is LitModules (e.g. `S2EFLitModule`) encapsulate
the training workflow, which takes an abstract GNN model and uses it to
evaluate whatever task. The GNN model is expected to be a subclass of
`AbstractTask` - specifically inheriting from either `AbstractS2EFModel` 
or `AbstractIS2REModel` (I guess sub-subclass).

The TL;DR would be:

class GraphConvModel(AbstractS2EFModel):
    ...
    def compute_energy(self, graph):
        ...

gnn = GraphConvModel(*args, **kwargs)

S2EFLitModule(
    gnn=gnn
)

"""
from typing import Any, Optional, Type, Union, Dict

from argparse import ArgumentParser
import numpy as np
import torch
import dgl
from dgl.nn import pytorch as dgl_nn
from torch import nn

from ocpmodels.models.base import AbstractDGLModel


class GraphConvBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_fc_layers: int = 3,
        fc_out_dim: Optional[int] = None,
        activation: Optional[Type[nn.Module]] = None,
    ) -> None:
        """
        Construct a single graph convolution interaction block, which comprises
        a graph convolution followed by sequential embedding transformation
        through linear layers.

        Parameters
        ----------
        in_dim : int
            Input dimensionality
        out_dim : int
            Output dimensionality of the convolution
        num_fc_layers : int, optional
            Number of fully-connected layers after convolution, by default 3
        fc_out_dim : Optional[int], optional
            Output dimensionality of the fully-connected transformations, by default None
            which defaults to `out_dim`
        activation : Optional[nn.Module], optional
            Reference to an activation function class, by default None
        """
        super().__init__()
        self.conv = dgl_nn.GraphConv(in_dim, out_dim, activation=activation())
        # if nothing is specified for the output MLP shape, just make it the same
        # as the number of message passing channels
        if fc_out_dim is None:
            fc_out_dim = out_dim
        self.fc_layers = self._make_layers(
            out_dim, fc_out_dim, num_fc_layers, activation
        )

    def forward(self, graph: dgl.DGLGraph, features: torch.Tensor) -> torch.Tensor:
        """
        Given a graph and node embeddings, perform graph convolution
        and sequentially transform the resulting node embeddings
        with linear layers.

        Parameters
        ----------
        graph : dgl.DGLGraph
            Molecular graph
        features : torch.Tensor
            Node embeddings

        Returns
        -------
        torch.Tensor
            Transformed node embeddings
        """
        n_z = self.conv(graph, features)
        n_z = self.fc_layers(n_z)
        return n_z

    @staticmethod
    def _make_layers(
        in_dim: int,
        out_dim: int,
        num_layers: int,
        activation: Optional[nn.Module] = None,
    ) -> Type[nn.Module]:
        layers = []
        sizes = np.linspace(in_dim, out_dim, num_layers).astype(int)
        for i in range(num_layers - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if activation is not None:
                layers.append(activation())
        return nn.Sequential(*layers)


class GraphConvModel(AbstractDGLModel):
    def __init__(
        self,
        atom_embedding_dim: int,
        out_dim: int,
        num_blocks: Optional[int] = 3,
        num_fc_layers: Optional[int] = 3,
        activation: Optional[Type[nn.Module]] = nn.SiLU,
        readout: Optional[Type[nn.Module]] = dgl_nn.SumPooling,
        num_atom_embedding: int = 100,
        embedding_kwargs: Dict[str, Any] = {},
        encoder_only: bool = True,
    ) -> None:
        super().__init__(
            atom_embedding_dim, num_atom_embedding, embedding_kwargs, encoder_only
        )
        """
        A simple baseline graph convolution model for use with energy/force
        regression. This model uses learnable atomic embeddings same as
        SchNet, and performs sequential graph convolution transformations
        to the node embeddings, perform graph pooling, and transforms the
        graph embeddings into an energy scalar.

        This class inherits from `AbstractS2EFModel` and so includes the
        force computation method, and only implements a concrete `compute_energy`
        method.

        Parameters
        ----------
        atom_embedding_dim : int
            Atomic embedding dimensionality
        out_dim : int
            Output dimensionality
        num_blocks : Optional[int], optional
            Number of convolution blocks, by default 3
        num_fc_layers : Optional[int], optional
            Number of fully-connected layers in each block, by default 3
        activation : Optional[nn.Module], optional
            Activation function to use between layers, by default nn.SiLU
        readout : Optional[nn.Module], optional
            Class to use for graph readout/pooling, by default dgl_nn.SumPooling
        """
        super().__init__()
        self.blocks = self._make_blocks(
            atom_embedding_dim + 3,
            out_dim,
            num_blocks,
            num_fc_layers,
            activation,
        )
        # if uninstantiated, create the pooling object
        if not isinstance(readout, nn.Module):
            readout = readout()
        self.readout = readout
        if not encoder_only:
            self.output = nn.Linear(out_dim, 1)

    def _forward(
        self,
        graph: dgl.DGLGraph,
        node_feats: torch.Tensor,
        pos: torch.Tensor,
        edge_feats: Optional[torch.Tensor] = None,
        graph_feats: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Implement the forward method, which computes the energy of
        a molecular graph.

        Parameters
        ----------
        graph : dgl.DGLGraph
            A single or batch of molecular graphs

        Returns
        -------
        torch.Tensor
            Energy tensor [G, 1] for G graphs
        """
        n_z = self.join_position_embeddings(pos, node_feats)
        with graph.local_scope():
            # recursively compress node embeddings, pool, and compute the energy
            for block in self.blocks:
                n_z = block(graph, n_z)
            output = self.readout(graph, n_z)
            if hasattr(self, "output"):
                # regress if we're not just an encoder
                output = self.output(output)
        return output

    @staticmethod
    def add_model_specific_args(
        parent_parser: Type[ArgumentParser],
    ) -> Type[ArgumentParser]:
        parser = parent_parser.add_argument_group("GraphConvModel")
        parser.add_argument(
            "--atom_embedding_dim",
            type=int,
            default=128,
            help="Atomic embeddings dimensionality",
        )
        parser.add_argument(
            "--out_dim",
            type=int,
            default=16,
            help="Output embeddings dimensionality.",
        )
        parser.add_argument(
            "--num_blocks",
            type=int,
            default=3,
            help="Number of graph convolution/interaction blocks.",
        )
        parser.add_argument(
            "--num_fc_layers",
            type=int,
            default=3,
            help="Number of fully-connected layers within convolution blocks.",
        )
        parser.add_argument(
            "--activation",
            type=str,
            default="nn.SiLU",
            help="String corresponding to the activation function class name. Must be in the launch script scope!",
        )
        parser.add_argument(
            "--readout",
            type=str,
            default="dgl_nn.SumPooling",
            help="String corresponds to the readout function class name. Must be in the launch script scope!",
        )
        return parent_parser

    @staticmethod
    def _make_blocks(
        in_dim: int,
        out_dim: int,
        num_blocks: Optional[int] = 3,
        num_fc_layers: Optional[int] = 3,
        activation: Optional[nn.Module] = None,
    ) -> nn.ModuleList:
        """
        Convenience static method for composing interaction blocks.

        Parameters
        ----------
        in_dim : int
            Input dimensionality, typically the embedding/feature dimensionality
        out_dim : int
            Output embedding dimensionality
        num_blocks : Optional[int], optional
            Number of convolution blocks, by default 3
        num_fc_layers : Optional[int], optional
            Number of linear layers following graph convolution, by default 3
        activation : Optional[nn.Module], optional
            Activation function to use between layers, by default None

        Returns
        -------
        nn.ModuleList
            A list of graph convolution interaction blocks
        """
        sizes = np.linspace(in_dim, out_dim, num_blocks).astype(int)
        blocks = []
        for depth in range(num_blocks - 1):
            blocks.append(
                GraphConvBlock(
                    sizes[depth],
                    sizes[depth + 1],
                    num_fc_layers,
                    activation=activation,
                )
            )
        return nn.ModuleList(blocks)
