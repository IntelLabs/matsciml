from __future__ import annotations

import pytest
import torch

from matsciml.models import losses


@pytest.fixture
def atom_weighted_l1():
    return losses.AtomWeightedL1()


@pytest.fixture
def atom_weighted_mse():
    return losses.AtomWeightedMSE()


@pytest.mark.parametrize(
    "shape",
    [
        (10,),
        (10, 3),
        (120, 1, 5),
    ],
)
def test_weighted_mse(atom_weighted_mse, shape):
    pred = torch.rand(*shape)
    target = torch.rand_like(pred)
    ptr = torch.randint(1, 100, (shape[0],))
    atom_weighted_mse(pred, target, ptr)


@pytest.mark.parametrize("shape", [(10,), (50, 3)])
@pytest.mark.parametrize(
    "quantiles",
    [
        {0.25: 0.5, 0.5: 1.0, 0.75: 3.21},
        {0.02: 0.2, 0.32: 0.67, 0.5: 1.0, 0.83: 2.0, 0.95: 10.0},
    ],
)
@pytest.mark.parametrize("use_norm", [True, False])
@pytest.mark.parametrize("loss_func", ["mse", "huber"])
def test_quantile_loss(shape, quantiles, use_norm, loss_func):
    # ensure we test against back prop as well
    x, y = torch.rand(2, *shape, requires_grad=True)
    l_func = losses.BatchQuantileLoss(quantiles, loss_func, use_norm, huber_delta=0.01)
    loss = l_func(x, y)
    loss.mean().backward()
