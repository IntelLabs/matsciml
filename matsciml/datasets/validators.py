from __future__ import annotations

from typing import Iterable

import numpy as np
import torch


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
