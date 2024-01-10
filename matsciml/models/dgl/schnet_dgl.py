# Copyright (C) 2022-3 Intel Corporation
# SPDX-License-Identifier: MIT License
from __future__ import annotations

from importlib.util import find_spec
from typing import Any, Dict, List, Optional, Type, Union

import dgl
import torch
from dgl.nn.pytorch import glob
from dgllife.model import SchNetGNN
from torch import nn

from matsciml.common.types import BatchDict, DataDict, Embeddings
from matsciml.models.base import AbstractDGLModel


class SchNet(AbstractDGLModel):
    def __init__(
        self,
        atom_embedding_dim: int,
        hidden_feats: list[int] | None = None,
        cutoff: float = 30.0,
        gap: float = 0.1,
        readout: type[nn.Module] | str | nn.Module = glob.AvgPooling,
        readout_kwargs: dict[str, Any] | None = None,
        num_atom_embedding: int = 100,
        embedding_kwargs: dict[str, Any] = {},
        encoder_only: bool = True,
    ) -> None:
        r"""
        Instantiate a stack of SchNet layers.

        This wrapper also comprises a readout function, and integrates into the
        matsciml pipeline with `encoder_only`.

        Parameters
        ----------
        atom_embedding_dim : int
            Dimensionality of the node embeddings
        hidden_feats : Optional[List[int]], default None
            Simultaneously sets the dimensionality of each SchNet layer
            and the number of layers by providing a list of ints. The
            default value is [64, 64, 64], i.e. three layers 64 wide each.
        num_atom_embedding : int
            Number of unique atom types
        cutoff : float
            Largest center in RBF expansion. Default to 30.
        gap : float
            Difference between two adjacent centers in RBF expansion. Default to 0.1.
        readout : Union[Type[nn.Module], str, nn.Module]
            Pooling function that aggregates node features after SchNet. You can
            specify either a reference to the pooling class directly, or an instance
            of a pooling operation. If a string is passed, we assume it refers to
            one of the glob functions implemented in DGL.
        readout_kwargs : Optional[Dict[str, Any]]
            Kwargs to pass into the construction of the readout function, if an
            instance was not passed
        encoder_only : bool
            Whether to return the graph embeddings only, and not return an
            energy value.

        """
        super().__init__(
            atom_embedding_dim,
            num_atom_embedding,
            embedding_kwargs,
            encoder_only,
        )
        self.model = SchNetGNN(
            atom_embedding_dim,
            hidden_feats,
            num_atom_embedding,
            cutoff,
            gap,
        )
        # copy over the embedding table to remove redundancy
        self.model.embed = self.atom_embedding
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
        if not hidden_feats:
            # based on default value from dgllife docs
            output_dim = 64
        else:
            output_dim = hidden_feats[-1]
        self.output = nn.Linear(output_dim, 1)

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
        data = {}
        graph = batch.get("graph")
        assert isinstance(
            graph,
            dgl.DGLGraph,
        ), f"Model {self.__class__.__name__} expects DGL graphs, but data in 'graph' key is type {type(graph)}"
        # SchNet expects atomic numbers as input to the model, so we do not
        # read from the embedding table like other models
        atomic_numbers = graph.ndata["atomic_numbers"].long()
        data["node_feats"] = atomic_numbers
        # extract interatomic distances
        assert (
            "r" in graph.edata
        ), f"SchNet expects interatomic distances as edge data under the 'r' key."
        data["edge_feats"] = graph.edata["r"]
        data["graph"] = graph
        data.setdefault("graph_feats", None)
        data.setdefault("pos", None)
        return data

    def _forward(
        self,
        graph: dgl.DGLGraph,
        node_feats: torch.Tensor,
        edge_feats: torch.Tensor,
        pos: torch.Tensor | None = None,
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
        edge_feats : torch.Tensor
            Tensor containing interatomic distances
        pos : Optional[torch.Tensor], optional
            XYZ coordinates of each atom, by default None and unused.
        graph_feats : Optional[torch.Tensor], optional
            Graph-based properties, by default None and unused.

        Returns
        -------
        Embeddings
            Data structure holding graph and node level embeddings.
        """
        n_z = self.model(graph, node_feats, edge_feats)
        g_z = self.readout(graph, n_z)
        return Embeddings(g_z, n_z)
