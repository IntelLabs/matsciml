# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: MIT License

from typing import Optional, Tuple
import math

from dgllife.model import SchNetGNN
from torch_cluster import radius_graph
import dgl
import torch
import ase
from torch import Tensor, nn
from dgl.nn.pytorch import CFConv

from ocpmodels.common.utils import (
    conditional_grad,
    get_pbc_distances,
    radius_graph_pbc,
)
from ocpmodels.models.base import AbstractS2EFModel


class SchNetDGLWrap(SchNetGNN):
    r"""Wrapper around the continuous-filter convolutional neural network SchNet from the
    `"SchNet: A Continuous-filter Convolutional Neural Network for Modeling
    Quantum Interactions" <https://arxiv.org/abs/1706.08566>`_. Each layer uses interaction
    block of the form:

    .. math::
        \mathbf{x}^{\prime}_i = \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j \odot
        h_{\mathbf{\Theta}} ( \exp(-\gamma(\mathbf{e}_{j,i} - \mathbf{\mu}))),

     SchNet-GNN Parameters
    ----------
    node_feats : int
        Size for node representations to learn. Default to 64.
    hidden_feats : list of int
        ``hidden_feats[i]`` gives the size of hidden representations for the i-th interaction
        layer. ``len(hidden_feats)`` equals the number of interaction layers.
        Default to ``[64, 64, 64]``.
    num_node_types : int
        Number of node types to embed. Default to 100.
    cutoff : float
        Largest center in RBF expansion. Default to 30.
    gap : float
        Difference between two adjacent centers in RBF expansion. Default to 0.1.
    """

    def __init__(
        self,
        num_atoms,  # not used
        bond_feat_dim,  # not used
        num_targets,
        use_pbc=True,
        regress_forces=True,
        otf_graph=False,
        node_feats=64,
        hidden_feats=128,
        num_node_types=128,
        cutoff=10.0,
        gap=0.1,
        readout="add",
    ):

        self.num_targets = num_targets
        self.regress_forces = regress_forces
        self.use_pbc = use_pbc
        self.cutoff = cutoff
        self.otf_graph = otf_graph

        self.num_node_types = num_node_types

        super(SchNetDGLWrap, self).__init__(
            node_feats=node_feats,
            hidden_feats=hidden_feats,
            num_node_types=num_node_types,
            cutoff=cutoff,
            gap=gap,
        )

    @conditional_grad(torch.enable_grad())
    def _forward(self, graph_data, label_data):

        z = graph_data.ndata["atomic_numbers"].long()

        pos = graph_data.ndata["pos"]

        edge_dists = torch.cdist(pos, pos)

        if self.otf_graph:
            edge_index, cell_offsets, neighbors = radius_graph_pbc(
                data, self.cutoff, 50
            )
            data.edge_index = edge_index
            data.cell_offsets = cell_offsets
            data.neighbors = neighbors
        energy = super(SchNetDGLWrap, self).forward(graph_data, z, edge_dists)

        return energy

    def forward(self, batch_data):

        graph_data, label_data = batch_data

        if self.regress_forces:
            graph_data.ndata["pos"].requires_grad_(True)
        energy = self._forward(graph_data, label_data)

        if self.regress_forces:
            forces = -1 * (
                torch.autograd.grad(
                    energy,
                    graph_data.ndata["pos"],
                    grad_outputs=torch.ones_like(energy),
                    create_graph=True,
                )[0]
            )
            return energy, forces
        else:
            return energy

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())


class PLSchNet(AbstractS2EFModel, SchNetDGLWrap):
    """
    Diamond inheritance; we want `SchNetDGLWrap`'s forward implementation,
    and the `AbstractS2EFModel`'s force computation and other general
    routines.
    """

    def __init__(
        self,
        num_targets,
        use_pbc=True,
        otf_graph=False,
        node_feats=64,
        hidden_feats=128,
        num_node_types=128,
        cutoff=10.0,
        gap=0.1,
    ):
        super().__init__(
            num_targets,
            use_pbc=use_pbc,
            regress_forces=False,
            otf_graph=otf_graph,
            node_feats=node_feats,
            hidden_feats=hidden_feats,
            num_node_types=num_node_types,
            cutoff=cutoff,
            gap=gap,
        )


class SchNet(AbstractS2EFModel):
    """
    This module implements the `SchNet` model using DGL abstractions. As a first pass,
    the aim is to make a best effort to port the existing model from PyTorch Geometric,
    however making some refactors to ensure flexibility for optimizations and reusability
    down the line.

    PyG
    CFConv(hidden_channels, num_gaussians, num_filters, cutoff)
    self.mlp = Sequential(
                Linear(num_gaussians, num_filters),
                ShiftedSoftplus(),
                Linear(num_filters, num_filters),
            )
            self.conv = CFConv(hidden_channels, hidden_channels, num_filters,
                            self.mlp, cutoff)
            self.act = ShiftedSoftplus()
            self.lin = Linear(hidden_channels, hidden_channels)

    DGL
    CFConv(node_in_feats, edge_in_feats, hidden_feats, out_feats)
    self.project_node = nn.Linear(node_in_feats, hidden_feats)
    self.project_edge = nn.Sequential(
                nn.Linear(edge_in_feats, hidden_feats),
                ShiftedSoftplus(),
                nn.Linear(hidden_feats, hidden_feats),
                ShiftedSoftplus()
            )
            self.project_out = nn.Sequential(
                nn.Linear(hidden_feats, out_feats),
                ShiftedSoftplus()
            )

    DGL -> PyG mapping
    edge_in_feats = num_gaussians
    hidden_feats = num_filters

    The intuition here is that the gaussians are mapped onto filters with `project_edge`

    The output of `project_out` can be different for DGL CFConv, whereas for PyG the
    input/output dimensions are equal to `hidden_channels`; for the same behavior we
    will set:
    out_feats = hidden_feats

    `project_out` is the output of the layer, similar to `lin` but with added softplus
    """

    def __init__(
        self,
        hidden_channels: int,
        num_filters: Optional[int] = 128,
        num_interactions: Optional[int] = 6,
        cutoff: Optional[float] = 10.0,
        num_gaussians: Optional[int] = 50,
        max_num_neighbors: Optional[int] = 32,
        readout_op: Optional[str] = "sum",
    ):
        super().__init__()
        # TODO this is not currently used, but presumably useful for embedding
        # lookup, so will need to revisit
        atomic_mass = torch.from_numpy(ase.data.atomic_masses)
        self.register_buffer("atomic_mass", atomic_mass)
        self.embedding = nn.Embedding(100, hidden_channels)
        self.rbf = RadialBasisExpansion(0, num_gaussians, cutoff, max_num_neighbors)
        # output layer of the model after interactions
        self.output = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            ShiftedSoftplus(),
            nn.Linear(hidden_channels // 2, 1),
        )
        # TODO make a comparison with the periodic embedding from the PyG implementation:
        # there is a mysterious cosine term within the PyG CFConv that is not in the SchNet paper.
        self.conv_layers = nn.ModuleList(
            [
                CFConv(
                    hidden_channels,
                    num_gaussians,
                    num_filters,
                    hidden_channels,
                )
                for _ in range(num_interactions)
            ]
        )
        self.readout_op = readout_op

    def compute_energy(self, graph: dgl.DGLGraph) -> Tensor:
        """
        Implements the model flow for SchNet to regress a molecular graph to energy.


        Parameters
        ----------
        graph : DGLGraph
            _description_

        Returns
        -------
        Tensor
            _description_
        """
        # maybe in the future store atomic numbers as longs instead
        # of having to downcast each time
        atoms = graph.ndata["atomic_numbers"].long()
        z = self.embedding(atoms)
        with graph.local_scope():
            # get distance based weights
            (edges, pairwise_distance, rbf_weights) = self.rbf(graph)
            # # TODO make sure that this step is needed; depends on if the distances
            # # are precomputed or not
            # # remove all edges from the graph, and re-add them based on the distances
            # if num_edges > 0:
            # # TODO This is breaking batching; adding new edges sets batch_size to 1
            # # and screws up the readout ops
            for interaction in self.conv_layers:
                # don't need to add it back onto itself as that's included in the
                # `update_all` step for CFConv
                z = interaction(graph, z, rbf_weights)
            # this generates a node-level embedding, with shape [N,1]
            graph.ndata["z"] = self.output(z)
            # graph pooling to get to shape [G,]
            energy = dgl.readout_nodes(graph, "z", op=self.readout_op)
            return energy


class RadialBasisExpansion(nn.Module):
    """
    Adapted from PyTorch Geometric source. This computes the RBF part of the interactions,
    including connectivity calculation and distance weighting with RBFs. For a large part,
    this is a refactor of the first part of `SchNet` in PyG.
    """

    def __init__(
        self,
        start: float = 0.0,
        num_gaussians: int = 50,
        cutoff: Optional[float] = 10.0,
        max_num_neighbors: Optional[int] = 32,
    ):
        super().__init__()
        centers = torch.linspace(start, cutoff, num_gaussians)
        self.spacing = -0.5 / (centers[1] - centers[0]).item() ** 2
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        self.register_buffer("centers", centers)

    @staticmethod
    def pos2edges(
        positions: Tensor,
        r: float,
        batch: Optional[Tensor] = None,
        max_num_neighbors: Optional[int] = 32,
    ):
        """
        Wrap the `radius_graph` function from `torch_cluster` for a good implementation of
        distance-based edge calculation.

        TODO find/implement similar operation in DGL to remove this dependency on `torch_cluster`.

        Parameters
        ----------
        positions : Tensor
            Tensor holding XYZ positions of each atom, shape [N, 3]
        r : float
            Radial distance cutoff for edge calculation
        batch : Optional[Tensor], optional
            Kwarg from torch_cluster, by default None
        max_num_neighbors : Optional[int], optional
            Kwarg from torch_cluster, by default 32

        Returns
        -------
        Tensor
            Edges between atoms; shape [2, num_edge]
        """
        return radius_graph(
            positions, r=r, batch=batch, max_num_neighbors=max_num_neighbors
        )

    def forward(self, graph: dgl.DGLGraph) -> Tuple[Tensor]:
        """
        Given a DGL graph of molecules, calculate the edges, distances, and
        radial basis functions weights. Assumes that the graph node data has key/value
        pair "xyz"/Tensor representing the atom coordinates.

        Parameters
        ----------
        graph : DGLGraph
            Input DGL graph

        Returns
        -------
        Tuple[Tensor]
            A three-tuple of Tensors, corresponding to: the edges [2, E] based
            off of the atom positions and cutoff; distances [E, 1]; radial basis
            function weights [E, G] for E edges, and G basis functions.
        """
        positions = graph.ndata["pos"]
        # here we need to use the `torch_cluster` method for associating each
        # node within a minibatch graph to its owner
        if graph.batch_size == 1:
            # if there's only one graph
            batch = torch.zeros(positions.size(0))
        else:
            batch_nodes = graph.batch_num_nodes()
            # TODO check that this is correctly generating the node indices
            batch = torch.cat(
                [torch.ones(length) * i for i, length in enumerate(batch_nodes)],
                dim=0,
            )
        with graph.local_scope():
            # unpack edge indices into source and destination nodes
            edges = self.pos2edges(
                positions, self.cutoff, batch, self.max_num_neighbors
            )
            src, dst = edges
            pairwise_distance = (positions[src] - positions[dst]).norm(dim=-1)
            # this computes the distance between a bond and an RBF center
            center_distances = pairwise_distance.view(-1, 1) - self.centers.view(1, -1)
            rbf_weights = (self.spacing * torch.pow(center_distances, 2)).exp()
            return (edges, pairwise_distance, rbf_weights)


class ShiftedSoftplus(torch.nn.Module):
    """
    Implements the ShiftedSoftplus (ssp) activation.
    """

    def __init__(self):
        super().__init__()
        self.shift = math.log(2.0)
        self.act = nn.Softplus()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(x) - self.shift
