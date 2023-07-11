# Copyright (C) 2022-3 Intel Corporation
# SPDX-License-Identifier: MIT License

from typing import Any, Union, Type, Optional, List, Dict
from importlib.util import find_spec

from dgllife.model import SchNetGNN
import dgl
import torch
from torch import nn
from dgl.nn.pytorch import glob

from ocpmodels.models.base import AbstractDGLModel


class SchNet(AbstractDGLModel):
    def __init__(
        self,
        atom_embedding_dim: int,
        hidden_feats: Optional[List[int]] = None,
        cutoff: float = 30.0,
        gap: float = 0.1,
        readout: Union[Type[nn.Module], str, nn.Module] = glob.AvgPooling,
        readout_kwargs: Optional[Dict[str, Any]] = None,
        num_atom_embedding: int = 100,
        embedding_kwargs: Dict[str, Any] = {},
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

        Raises
        ------
        ImportError:
            [TODO:description]
        """
        super().__init__(
            atom_embedding_dim, num_atom_embedding, embedding_kwargs, encoder_only
        )
        self.model = SchNetGNN(
            atom_embedding_dim, hidden_feats, num_atom_embedding, cutoff, gap
        )
        # copy over the embedding table to remove redundancy
        self.model.embed = self.atom_embedding
        if isinstance(readout, (str, Type)):
            # if str, assume it's the name of a class
            if isinstance(readout, str):
                readout_cls = find_spec(readout, "dgl.nn.pytorch.glob")
                if readout_cls is None:
                    raise ImportError(f"Class name passed to `readout`, but not found in `dgl.nn.pytorch.glob`.")
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
            raise ValueError(f"No graph passed, and `graph` key does not exist in batch.")
        # extract out node and edge features as expected
        node_feats = graph.ndata.get("atomic_numbers").long()
        edge_feats = graph.edata.get("r", None)
        if edge_feats is None:
            raise ValueError("`r` key is missing from graph edge data. Please use the `DistancesTransform`.")
        # run through the model
        n_z = self.model(graph, node_feats, edge_feats)
        g_z = self.readout(graph, n_z)
        if self.encoder_only:
            return g_z
        return self.output(g_z)
