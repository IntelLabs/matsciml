# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: MIT License

from typing import Optional
import torch
import torch.nn as nn
import dgl

from ocpmodels.models.dgl.dpp import dimenet_utils as du
from ocpmodels.models.base import AbstractEnergyModel

"""
Credit for original code: xnuohz; https://github.com/xnuohz/DimeNet-dgl
"""


class DimeNetPP(AbstractEnergyModel):
    """
    DimeNet++ model.
    Parameters
    ----------
    emb_size
        Embedding size used for the messages
    out_emb_size
        Embedding size used for atoms in the output block
    int_emb_size
        Embedding size used for interaction triplets
    basis_emb_size
        Embedding size used inside the basis transformation
    num_blocks
        Number of building blocks to be stacked
    num_spherical
        Number of spherical harmonics
    num_radial
        Number of radial basis functions
    cutoff
        Cutoff distance for interatomic interactions
    envelope_exponent
        Shape of the smooth cutoff
    num_before_skip
        Number of residual layers in interaction block before skip connection
    num_after_skip
        Number of residual layers in interaction block after skip connection
    num_dense_output
        Number of dense layers for the output blocks
    num_targets
        Number of targets to predict
    activation
        Activation function
    extensive
        Whether the output should be extensive (proportional to the number of atoms)
    output_init
        Initial function in output block
    """

    def __init__(
        self,
        emb_size: Optional[int] = 128,
        out_emb_size: Optional[int] = 256,
        int_emb_size: Optional[int] = 64,
        basis_emb_size: Optional[int] = 8,
        num_blocks: Optional[int] = 4,
        num_spherical: Optional[int] = 7,
        num_radial: Optional[int] = 6,
        cutoff: Optional[float] = 5.0,
        envelope_exponent: Optional[float] = 5.0,
        num_before_skip: Optional[int] = 1,
        num_after_skip: Optional[int] = 2,
        num_dense_output: Optional[int] = 3,
        activation: Optional[nn.Module] = nn.SiLU,
        extensive: Optional[bool] = True,
    ):
        super(DimeNetPP, self).__init__()

        self.num_blocks = num_blocks
        self.num_radial = num_radial

        # cosine basis function expansion layer
        self.rbf_layer = du.BesselBasisLayer(
            num_radial=num_radial,
            cutoff=cutoff,
            envelope_exponent=envelope_exponent,
        )

        self.sbf_layer = du.SphericalBasisLayer(
            num_spherical=num_spherical,
            num_radial=num_radial,
            cutoff=cutoff,
            envelope_exponent=envelope_exponent,
        )

        # embedding block
        self.emb_block = du.EmbeddingBlock(
            emb_size=emb_size,
            num_radial=num_radial,
            bessel_funcs=self.sbf_layer.get_bessel_funcs(),
            cutoff=cutoff,
            envelope_exponent=envelope_exponent,
            activation=activation,
        )

        # output block
        self.output_blocks = nn.ModuleList(
            {
                du.OutputPPBlock(
                    emb_size=emb_size,
                    out_emb_size=out_emb_size,
                    num_radial=num_radial,
                    num_dense=num_dense_output,
                    num_targets=1,
                    activation=activation,
                    extensive=extensive,
                )
                for _ in range(num_blocks + 1)
            }
        )

        # interaction block
        self.interaction_blocks = nn.ModuleList(
            {
                du.InteractionPPBlock(
                    emb_size=emb_size,
                    int_emb_size=int_emb_size,
                    basis_emb_size=basis_emb_size,
                    num_radial=num_radial,
                    num_spherical=num_spherical,
                    num_before_skip=num_before_skip,
                    num_after_skip=num_after_skip,
                    activation=activation,
                )
                for _ in range(num_blocks)
            }
        )
        self.save_hyperparameters()

    def edge_init(self, edges):
        # Calculate angles k -> j -> i
        R1, R2 = edges.src["o"], edges.dst["o"]
        x = torch.sum(R1 * R2, dim=-1)
        y = torch.cross(R1, R2)
        y = torch.norm(y, dim=-1)
        angle = torch.atan2(y, x)
        # Transform via angles
        cbf = [f(angle) for f in self.sbf_layer.get_sph_funcs()]
        cbf = torch.stack(cbf, dim=1)  # [None, 7]
        cbf = cbf.repeat_interleave(self.num_radial, dim=1)  # [None, 42]
        # Notice: it's dst, not src
        sbf = edges.dst["rbf_env"] * cbf  # [None, 42]
        return {"sbf": sbf}

    @staticmethod
    def _create_line_graph(graph: dgl.DGLGraph) -> dgl.DGLGraph:
        if graph.batch_size == 1:
            return dgl.line_graph(graph, backtracking=False)
        else:
            # in the case we have multiple graphs, unbatch and
            # create line graphs then
            graphs = dgl.unbatch(graph)
            l_g = [dgl.line_graph(g, backtracking=False) for g in graphs]
            return dgl.batch(l_g)

    @staticmethod
    def edge_distance(graph: dgl.DGLGraph) -> None:
        """
        Compute edge distances on the fly; applies a lambda function
        that computes the Euclidean distance between two atoms, and
        sets the edge property "r".

        Parameters
        ----------
        graph : dgl.DGLGraph
            Input molecular graph, assuming atomic positions are
            stored in the node data "pos"
        """
        src, dst = graph.edges()
        pos = graph.ndata["pos"]
        src_pos, dst_pos = pos[src], pos[dst]
        graph.edata["r"] = nn.functional.pairwise_distance(src_pos, dst_pos, p=2.0)
        graph.edata["o"] = src_pos - dst_pos
        return graph

    def forward(self, g: dgl.DGLGraph):
        g = self.edge_distance(g)
        l_g = self._create_line_graph(g)
        # add rbf features for each edge in one batch graph, [num_radial,]
        g = self.rbf_layer(g)
        # Embedding block
        g = self.emb_block(g)
        # Output block
        P = self.output_blocks[0](g)  # [batch_size, num_targets]
        # Prepare sbf feature before the following blocks
        for k, v in g.edata.items():
            l_g.ndata[k] = v

        l_g.apply_edges(self.edge_init)
        # Interaction blocks
        for i in range(self.num_blocks):
            g = self.interaction_blocks[i](g, l_g)
            P += self.output_blocks[i + 1](g)
        return P
