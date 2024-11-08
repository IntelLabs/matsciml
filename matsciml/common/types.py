from __future__ import annotations

from typing import Callable, Union

import torch
from pydantic import ConfigDict, Field, ValidationError, field_validator, BaseModel

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


class Embeddings(BaseModel):
    """
    Data structure that packs together embeddings from a model.
    """

    system_embedding: torch.Tensor | None = None
    point_embedding: torch.Tensor | None = None
    reduction: str | Callable | None = None
    reduction_kwargs: dict[str, str | float] = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)

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


class ModelOutput(BaseModel):
    """
    Standardized output data structure out of models.

    The advantage of doing is to standardize keys, as well
    as to standardize shapes the are produced by models;
    i.e. remove unused dimensions using ``pydantic``
    validation mechanisms.
    """

    batch_size: int
    embeddings: Embeddings | None = None
    node_energies: torch.Tensor | None = None
    total_energy: torch.Tensor | None = None
    forces: torch.Tensor | None = None
    stresses: torch.Tensor | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("total_energy", mode="before")
    @classmethod
    def standardize_total_energy(cls, values: torch.Tensor) -> torch.Tensor:
        """
        Check to ensure the total energy tensor being passed
        is ultimately scalar.

        Parameters
        ----------
        values : torch.Tensor
            Tensor holding energy values for each graph/system
            within a batch.

        Returns
        -------
        torch.Tensor
            1-D tensor containing energies for each graph/system
            within a batch.

        Raises
        ------
        ValidationError:
            If after running ``squeeze`` on the input tensor, the
            dimensions are still greater than one we raise a
            ``ValidationError``.
        """
        # drop all redundant dimensions
        values = values.squeeze()
        # last step is an assertion check for QA
        if values.ndim != 1:
            raise ValidationError(
                f"Expected graph/system energies to be scalar; got shape {values.shape}"
            )
        return values

    @field_validator("forces", mode="after")
    @classmethod
    def check_force_shape(cls, forces: torch.Tensor) -> torch.Tensor:
        """
        Check to ensure that the force tensor has the expected
        shape. Runs after the type checking by ``pydantic``.

        Parameters
        ----------
        forces : torch.Tensor
            Force tensor to check.

        Returns
        -------
        torch.Tensor
            Validated force tensor without modifications.

        Raises
        ------
        ValidationError:
            If the dimensionality of the tensor is not 2D, and/or
            if the last dimensionality of the tensor is not 3-long.
        """
        if forces.ndim != 2:
            raise ValidationError(
                f"Expected force tensor to be 2D; got {forces.shape}."
            )
        if forces.size(-1) != 3:
            raise ValidationError(
                f"Expected last dimension of forces to be length 3; got {forces.shape}."
            )
        return forces
