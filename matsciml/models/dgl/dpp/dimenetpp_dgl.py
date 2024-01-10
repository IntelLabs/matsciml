# Copyright (C) 2022-3 Intel Corporation
# SPDX-License-Identifier: MIT License
from __future__ import annotations

from typing import Any, Dict, Optional

import dgl
import torch
import torch.nn as nn

from matsciml.models.base import AbstractDGLModel
from matsciml.models.dgl.dpp import dimenet_utils as du

"""
Credit for original code: xnuohz; https://github.com/xnuohz/DimeNet-dgl
"""


class DimeNetPP(AbstractDGLModel):
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
        emb_size: int | None = 128,
        out_emb_size: int | None = 256,
        int_emb_size: int | None = 64,
        basis_emb_size: int | None = 8,
        num_blocks: int | None = 4,
        num_spherical: int | None = 7,
        num_radial: int | None = 6,
        cutoff: float | None = 5.0,
        envelope_exponent: float | None = 5.0,
        num_before_skip: int | None = 1,
        num_after_skip: int | None = 2,
        num_dense_output: int | None = 3,
        activation: nn.Module | None = nn.SiLU,
        extensive: bool | None = True,
        num_atom_embedding: int = 100,
        atom_embedding_dim: int | None = None,
        embedding_kwargs: dict[str, Any] = {},
        num_targets: int | None = None,
        encoder_only: bool = True,
    ) -> None:
        if atom_embedding_dim:
            raise ValueError(
                f"'atom_embedding_dim' should not be specified; please pass 'emb_size' instead.",
            )
        super().__init__(emb_size, num_atom_embedding, embedding_kwargs, encoder_only)
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
        # overwrite the redundant embedding table
        self.emb_block.embedding = self.atom_embedding

        # output block
        self.output_blocks = nn.ModuleList(
            {
                du.OutputPPBlock(
                    emb_size=emb_size,
                    out_emb_size=out_emb_size,
                    num_radial=num_radial,
                    num_dense=num_dense_output,
                    num_targets=num_targets,
                    activation=activation,
                    extensive=extensive,
                    encoder_only=encoder_only,
                )
                for _ in range(num_blocks + 1)
            },
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
            },
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
    def edge_distance(
        graph: dgl.DGLGraph,
        pos: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
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
        src_pos, dst_pos = pos[src], pos[dst]
        r = nn.functional.pairwise_distance(src_pos, dst_pos, p=2.0)
        o = src_pos - dst_pos
        return {"r": r, "o": o}

    def _forward(
        self,
        graph: dgl.DGLGraph,
        node_feats: torch.Tensor,
        pos: torch.Tensor,
        edge_feats: torch.Tensor | None = None,
        graph_feats: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
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
        torch.Tensor
            Graph embeddings, or output value if not 'encoder_only'
        """
        dist_dict = self.edge_distance(graph, pos)
        with graph.local_scope():
            for key in ["r", "o"]:
                graph.edata[key] = dist_dict[key]
            l_g = self._create_line_graph(graph)
            # add rbf features for each edge in one batch graph, [num_radial,]
            graph.edata["rbf"] = self.rbf_layer(dist_dict["r"])
            # Embedding block
            graph = self.emb_block(graph, node_feats)
            # Output block
            P = self.output_blocks[0](graph)
            # Prepare sbf feature before the following blocks
            for k, v in graph.edata.items():
                l_g.ndata[k] = v

            l_g.apply_edges(self.edge_init)
            # Interaction blocks
            for i in range(self.num_blocks):
                graph = self.interaction_blocks[i](graph, l_g)
                P += self.output_blocks[i + 1](graph)
        return P
