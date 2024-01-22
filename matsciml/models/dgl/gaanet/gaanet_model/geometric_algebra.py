# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: MIT License
from __future__ import annotations

import functools

import torch

# def custom_norm(inputs: torch.Tensor) -> torch.Tensor:
#


@functools.lru_cache
def _bivec_dual_swizzle(dtype, device):
    swizzle = torch.as_tensor(
        [[0, 0, 0, -1], [0, 0, 1, 0], [0, -1, 0, 0], [1, 0, 0, 0]],
        dtype=dtype,
        device=device,
    )
    return swizzle


@functools.lru_cache
def _vecvec_swizzle(dtype, device):
    # 0 1 2
    # 3 4 5
    # 6 7 8
    swizzle = torch.as_tensor(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, -1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, -1, 0],
            [0, 0, 0, -1],
            [1, 0, 0, 0],
        ],
        dtype=dtype,
        device=device,
    )
    return swizzle


@functools.lru_cache
def _bivecvec_swizzle(dtype, device):
    # 0 1 2
    # 3 4 5
    # 6 7 8
    # 9 10 11
    swizzle = torch.as_tensor(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, -1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, -1, 0],
            [0, 0, 0, -1],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, -1, 0],
            [0, 1, 0, 0],
        ],
        dtype=dtype,
        device=device,
    )
    return swizzle


@functools.lru_cache
def _trivecvec_swizzle(dtype, device):
    # 0 1 2
    # 3 4 5
    # 6 7 8
    # 9 10 11
    swizzle = torch.as_tensor(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, -1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, -1, 0],
            [0, 0, 0, -1],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, -1, 0],
            [0, 1, 0, 0],
        ],
        dtype=dtype,
        device=device,
    )
    return swizzle


class CustomNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        result = torch.linalg.norm(x, axis=-1, keepdims=True)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        y = custom_norm(x)
        eps = torch.as_tensor(1e-19, dtype=x.dtype, device=x.device)
        return grad_output * (x / torch.maximum(y, eps))


custom_norm = CustomNorm.apply


def calculate_products(
    inputs_a: torch.Tensor,
    inputs_b: torch.Tensor,
    last_dim: int,
) -> torch.Tensor:
    result = inputs_a.unsqueeze(-1) * inputs_b.unsqueeze(-2)

    new_shape = list(result.shape[:-2])
    new_shape.append(last_dim)

    result = result.reshape(new_shape)

    return result


def bivector_dual(inputs: torch.Tensor) -> torch.Tensor:
    """scalar + bivector -> vector + trivector

    Calculates the dual of an input value, expressed as (scalar, bivector)
    with basis (1, e12, e13, e23)."""

    swizzle = _bivec_dual_swizzle(inputs.dtype, inputs.device).detach()

    result = torch.tensordot(inputs, swizzle, dims=1)

    return result


def vector_vector(vector_a: torch.Tensor, vector_b: torch.Tensor) -> torch.Tensor:
    """vector * vector -> scalar + bivector

    Calculates the product of two vector inputs with basis (e1, e2, e3).
    Produces a (scalar, bivector) output with basis (1, e12, e13, e23)."""

    products = calculate_products(vector_a, vector_b, 9)

    swizzle = _vecvec_swizzle(products.dtype, products.device).detach()

    result = torch.tensordot(products, swizzle, dims=1)

    return result


def vector_vector_invariants(inputs: torch.Tensor) -> torch.Tensor:
    """Calculates rotation-invariant attributes of a (scalar, bivector) quantity.

    Returns a 2D output: the scalar and norm of the bivector."""

    result = torch.cat([inputs[..., :1], custom_norm(inputs[..., 1:4])], dim=-1)

    return result


def vector_vector_covariants(inputs: torch.Tensor) -> torch.Tensor:
    """Calculates rotation-covariant attributes of a (scalar, bivector) quantity.

    Converts the bivector to a vector by taking the dual."""

    result = bivector_dual(inputs)[..., :3]

    return result


def bivector_vector(bivector: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """(scalar + bivector) * vector -> vector + trivector

    Calculates the product of a (scalar + bivector) and a vector. The two inputs
    are expressed in terms of the basis (1, e12, e13, e23) and (e1, e2, e3);
    the output is expressed in terms of the basis (e1, e2, e3, e123)."""

    products = calculate_products(bivector, vector, 12)

    swizzle = _bivecvec_swizzle(dtype=products.dtype, device=products.device)

    result = torch.tensordot(products, swizzle, dims=1)

    return result


def bivector_vector_invariants(inputs: torch.Tensor) -> torch.Tensor:
    """Calculates rotation-invariant attributes of a (vector, trivector) quantity.

    Returns a 2D output: the norm of the vector and the trivector."""

    result = torch.cat([custom_norm(inputs[..., :3]), inputs[..., 3:4]], dim=-1)

    return result


def bivector_vector_covariants(inputs: torch.Tensor) -> torch.Tensor:
    """Calculates rotation-covariant attributes of a (vector, trivector) quantity.

    Returns the vector."""

    result = inputs[..., :3]

    return result


def trivector_vector(trivector: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """(vector + trivector) * vector -> scalar + bivector

    Calculates the product of a (vector + trivector) and a vector. The two
    inputs are expressed in terms of the basis (e1, e2, e3, e123) and
    (e1, e2, e3); the output is expressed in terms of the basis
    (1, e12, e13, e23)."""

    products = calculate_products(trivector, vector, 12)

    swizzle = _trivecvec_swizzle(dtype=products.dtype, device=products.device)

    result = torch.tensordot(products, swizzle, dims=1)

    return result


trivector_vector_invariants = vector_vector_invariants
trivector_vector_covariants = vector_vector_covariants
