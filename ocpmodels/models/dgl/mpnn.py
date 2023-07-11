# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: MIT License

from typing import Any, Union, Type, Optional, List, Dict
from importlib.util import find_spec

from dgllife.model import MPNNGNN
import dgl
import torch
from torch import nn
from dgl.nn.pytorch import glob

from ocpmodels.models.base import AbstractDGLModel


class MPNN(AbstractDGLModel):
    def __init__(
        self,
        atom_embedding_dim: int,
        edge_in_dim: int = 1,
        node_out_dim: int = 64,
        edge_out_dim: int = 128,
        num_step_message_passing: int = 3,
        num_atom_embedding: int = 100,
        embedding_kwargs: Dict[str, Any] = ...,
        readout: Union[Type[nn.Module], str, nn.Module] = glob.AvgPooling,
        readout_kwargs: Optional[Dict[str, Any]] = None,
        encoder_only: bool = True,
    ) -> None:
        """
        Instantiate a stack of SchNet layers.

        This wrapper also comprises a readout function, and integrates into the
        matsciml pipeline with `encoder_only`.

        Parameters
        ----------
        node_embedding_dim : int
            Dimensionality of the node embeddings
        edge_in_dim : int
            Dimensionality of the edge features; default one for pairwise distance
        node_out_dim : int
            Dimensionality of the node embeddings after message passing
        edge_out_dim : int
            Dimensionality of the hidden edge features
        num_step_message_passing : int
            Number of message passing steps
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

        Raises
        ------
        ImportError:
            [TODO:description]
        """
        super().__init__(
            atom_embedding_dim, num_atom_embedding, embedding_kwargs, encoder_only
        )
        self.model = MPNNGNN(
            atom_embedding_dim + 3,
            edge_in_dim,
            node_out_dim,
            edge_out_dim,
            num_step_message_passing,
        )
        if isinstance(readout, (str, Type)):
            # if str, assume it's the name of a class
            if isinstance(readout, str):
                readout_cls = find_spec(readout, "dgl.nn.pytorch.glob")
                if readout_cls is None:
                    raise ImportError(
                        f"Class name passed to `readout`, but not found in `dgl.nn.pytorch.glob`."
                    )
            else:
                # assume it's generic type
                readout_cls = readout
            if readout_kwargs is None:
                readout_kwargs = {}
            readout = readout_cls(**readout_kwargs)
        self.readout = readout
        self.encoder_only = encoder_only
        self.output = nn.Linear(node_out_dim, 1)

    def forward(
        self,
        batch: Optional[
            Dict[str, Union[torch.Tensor, dgl.DGLGraph, Dict[str, torch.Tensor]]]
        ] = None,
        graph: Optional[dgl.DGLGraph] = None,
    ) -> torch.Tensor:
        if batch is not None:
            graph = batch.get("graph", None)
        if not graph and not batch:
            raise ValueError(
                f"No graph passed, and `graph` key does not exist in batch."
            )
        # grab atom numbers and expand into learned embedding
        node_feats = graph.ndata.get("atomic_numbers").long()
        node_feats = self.embedding(node_feats)
        edge_feats = graph.edata.get("r", None)
        if edge_feats is None:
            raise ValueError(
                "`r` key is missing from graph edge data. Please use the `DistancesTransform`."
            )
        # run through the model
        n_z = self.model(graph, node_feats, edge_feats)
        g_z = self.readout(graph, n_z)
        if self.encoder_only:
            return g_z
        return self.output(g_z)
