###########################################################################################
# Implementation of MACE models and other models based E(3)-Equivariant MPNNs
# (https://github.com/ACEsuit/mace)
# Original Authors: Ilyes Batatia, Gregor Simm
# Integrated into matsciml by Vaibhav Bihani, Sajid Mannan
# Refactors and improved docstrings by Kelvin Lee
# This program is distributed under the MIT License
###########################################################################################
"""basic scatter_sum operations from torch_scatter from
https://github.com/mir-group/pytorch_runstats/blob/main/torch_runstats/scatter_sum.py
Using code from https://github.com/rusty1s/pytorch_scatter, but cut down to avoid a dependency.
PyTorch plans to move these features into the main repo, but until then,
to make installation simpler, we need this pure python set of wrappers
that don't require installing PyTorch C++ extensions.
See https://github.com/pytorch/pytorch/issues/63780.
"""

from __future__ import annotations

from typing import Optional

import torch

__all__ = ["scatter_sum", "scatter_std", "scatter_mean"]


def _broadcast(src: torch.Tensor, other: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Broadcasts ``src`` to yield a tensor with equivalent shape to ``other``
    along dimension ``dim``.

    Parameters
    ----------
    src : torch.Tensor
        Tensor to broadcast into a new shape.
    other : torch.Tensor
        Tensor to match shape against.
    dim : int
        Dimension to broadcast values along.

    Returns
    -------
    torch.Tensor
        Broadcasted values of ``src``, with the same shape as ``other``.
    """
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand_as(other)
    return src


@torch.jit.script
def scatter_sum(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: torch.Tensor | None = None,
    dim_size: int | None = None,
    reduce: str = "sum",
) -> torch.Tensor:
    """
    Apply a scatter operation with sum reduction, from ``src``
    to ``out`` at indices ``index`` along the specified
    dimension.

    The function will apply a ``_broadcast`` with ``index``
    to reshape it to the same as ``src`` first, then allocate
    a new tensor based on the expected final shape (depending
    on ``dim``).

    Parameters
    ----------
    src : torch.Tensor
        Tensor containing source values to scatter add.
    index : torch.Tensor
        Indices for the scatter add operation.
    dim : int, optional
        Dimension to apply the scatter add operation, by default -1
    out : torch.Tensor, optional
        Output tensor to store the scatter sum result, by default None,
        which will create a tensor with the correct shape within
        this function.
    dim_size : int, optional
        Used to determine the output shape, by default None, which
        will then infer the output shape from ``dim``.
    reduce : str, optional
        Unused and kept for backwards compatibility.

    Returns
    -------
    torch.Tensor
        Resulting scatter sum output.
    """
    assert reduce == "sum"  # for now, TODO
    index = _broadcast(index, src, dim)
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        return out.scatter_add_(dim, index, src)
    else:
        return out.scatter_add_(dim, index, src)


@torch.jit.script
def scatter_std(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
    unbiased: bool = True,
) -> torch.Tensor:
    if out is not None:
        dim_size = out.size(dim)

    if dim < 0:
        dim = src.dim() + dim

    count_dim = dim
    if index.dim() <= dim:
        count_dim = index.dim() - 1

    ones = torch.ones(index.size(), dtype=src.dtype, device=src.device)
    count = scatter_sum(ones, index, count_dim, dim_size=dim_size)

    index = _broadcast(index, src, dim)
    tmp = scatter_sum(src, index, dim, dim_size=dim_size)
    count = _broadcast(count, tmp, dim).clamp(1)
    mean = tmp.div(count)

    var = src - mean.gather(dim, index)
    var = var * var
    out = scatter_sum(var, index, dim, out, dim_size)

    if unbiased:
        count = count.sub(1).clamp_(1)
    out = out.div(count + 1e-6).sqrt()

    return out


@torch.jit.script
def scatter_mean(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
) -> torch.Tensor:
    out = scatter_sum(src, index, dim, out, dim_size)
    dim_size = out.size(dim)

    index_dim = dim
    if index_dim < 0:
        index_dim = index_dim + src.dim()
    if index.dim() <= index_dim:
        index_dim = index.dim() - 1

    ones = torch.ones(index.size(), dtype=src.dtype, device=src.device)
    count = scatter_sum(ones, index, index_dim, None, dim_size)
    count[count < 1] = 1
    count = _broadcast(count, out, dim)
    if out.is_floating_point():
        out.true_divide_(count)
    else:
        out.div_(count, rounding_mode="floor")
    return out
