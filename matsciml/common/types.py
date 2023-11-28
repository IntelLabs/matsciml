from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Union

import torch

from matsciml.common import package_registry

# for point clouds
representations = [torch.Tensor]
graph_types = []

if package_registry["pyg"]:
    from torch_geometric.data import Data as PyGGraph

    representations.append(PyGGraph)
    graph_types.append(PyGGraph)
if package_registry["dgl"]:
    from dgl import DGLGraph

    representations.append(DGLGraph)
    graph_types.append(DGLGraph)

ModelingTypes = tuple(representations)
GraphTypes = tuple(graph_types)

DataType = Union[ModelingTypes]
AbstractGraph = Union[GraphTypes]

# for a dictionary look up of data
DataDict = dict[str, Union[float, DataType]]

# for a dictionary of batched data
BatchDict = dict[str, Union[float, DataType, DataDict]]


@dataclass
class Embeddings:
    """
    Data structure that packs together embeddings from a model.
    """

    system_embedding: torch.Tensor | None = None
    point_embedding: torch.Tensor | None = None
    reduction: str | Callable | None = None
    reduction_kwargs: dict[str, str | float] = field(default_factory=dict)

    @property
    def num_points(self) -> int:
        if not isinstance(self.point_embedding, torch.Tensor):
            raise ValueError("No point-level embeddings stored!")
        return self.point_embedding.size(0)

    @property
    def batch_size(self) -> int:
        if not isinstance(self.system_embedding, torch.Tensor):
            raise ValueError(
                "No system-level embeddings stored, can't determine batch size!",
            )
        return self.system_embedding.size(0)

    def reduce_point_embeddings(
        self,
        reduction: str | Callable | None = None,
        **reduction_kwargs,
    ) -> torch.Tensor:
        """
        Perform a reduction/readout of the point-level embeddings to obtain
        system/graph-level embeddings.

        This function provides a regular interface for obtaining system-level
        embeddings by either passing a function that functions via:

        ``system_level = reduce(point_level)``

        or by passing a ``str`` name of a function from ``torch`` such as ``mean``.
        """
        assert isinstance(
            self.point_embedding,
            torch.Tensor,
        ), "No point-level embeddings stored to reduce."
        if not reduction:
            reduction = self.reduction
        if isinstance(reduction, str):
            reduction = getattr(torch, reduction)
        if not reduction:
            raise ValueError("No method for reduction passed.")
        self.reduction_kwargs.update(reduction_kwargs)
        system_embeddings = reduction(self.point_embedding, **self.reduction_kwargs)
        self.system_embedding = system_embeddings
        return system_embeddings
