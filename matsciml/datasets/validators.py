from __future__ import annotations

from typing_extensions import Annotated
from typing import Iterable

from ase.geometry import complete_cell
import numpy as np
import torch

from pydantic import BeforeValidator, AfterValidator, PlainSerializer


def coerce_long_like(data: torch.Tensor) -> torch.Tensor:
    """If the input tensor is not integer type, cast to int64"""
    if torch.is_floating_point(data):
        return data.long()
    return data


def coerce_float_like(data: torch.Tensor) -> torch.Tensor:
    """If the input tensor is not floating point, cast to fp32"""
    if not torch.is_floating_point(data):
        return data.float()
    return data


def array_like_serialization(data: np.ndarray | torch.Tensor) -> list:
    """Map array-like data to list for JSON serialization"""
    return data.tolist()


def cast_to_torch(data: float | int | Iterable[float | int]) -> torch.Tensor:
    """
    Coerces data into PyTorch tensors when possible. Assumes only
    numeric data is passed into this function (e.g. strings will fail)

    Parameters
    ----------
    data : Any
        Coerces iterable data into tensors. For NumPy arrays,
        we use `torch.from_numpy`. Scalar values are also
        packed into tensors.

    Returns
    -------
    torch.Tensor
        A tensor with matching data type and shape.
    """
    # wrap as a list to use the same pipeline
    if isinstance(data, float | int):
        data = [data]
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    else:
        return torch.tensor(data)


def check_coord_dims(data: torch.Tensor) -> torch.Tensor:
    """For coordinate tensors, check to make sure the last dimension is 3D"""
    assert data.size(-1) == 3, "Last dimension of coordinate tensors should be size 3."
    return data


def check_lattice_matrix_like(data: torch.Tensor) -> torch.Tensor:
    if data.ndim < 2:
        raise ValueError("Lattice matrix should be at least 2D")
    last_dims = data.shape[-2:]
    if last_dims != (3, 3):
        raise ValueError("Lattice matrix should be (3, 3) in last two dimensions.")
    return data


def check_lattice_ortho(data: torch.Tensor) -> torch.Tensor:
    """Check if the lattice matrix comprises orthogonal basis vectors"""
    # recasts after check
    return torch.from_numpy(complete_cell(data))


def check_edge_like(data: torch.Tensor) -> torch.Tensor:
    """Check if a tensor resembles expected edge indices"""
    if data.ndim != 2:
        raise ValueError("Edge tensor should be 2D")
    if data.size(0) != 2:
        raise ValueError("First dimension of edge tensor should be shape 2.")
    if data.dtype != torch.long:
        raise ValueError("Edge indices should be long type.")
    return data


# type for coordinate-like tensors
CoordTensor = Annotated[
    torch.Tensor,
    BeforeValidator(cast_to_torch),
    AfterValidator(check_coord_dims),
    AfterValidator(coerce_float_like),
    PlainSerializer(array_like_serialization),
]

Float1DTensor = Annotated[
    torch.Tensor,
    BeforeValidator(cast_to_torch),
    AfterValidator(coerce_float_like),
    PlainSerializer(array_like_serialization),
]

Long1DTensor = Annotated[
    torch.Tensor,
    BeforeValidator(cast_to_torch),
    AfterValidator(coerce_long_like),
    PlainSerializer(array_like_serialization),
]

# reuses the lattice matrix check which is functionally the same
StressTensor = Annotated[
    torch.Tensor,
    BeforeValidator(cast_to_torch),
    AfterValidator(check_lattice_matrix_like),
    AfterValidator(coerce_float_like),
    PlainSerializer(array_like_serialization),
]

LatticeTensor = Annotated[
    torch.Tensor,
    BeforeValidator(cast_to_torch),
    AfterValidator(check_lattice_matrix_like),
    AfterValidator(coerce_float_like),
    PlainSerializer(array_like_serialization),
]

EdgeTensor = Annotated[
    torch.Tensor,
    BeforeValidator(cast_to_torch),
    AfterValidator(coerce_long_like),
    AfterValidator(check_edge_like),
    PlainSerializer(array_like_serialization),
]
