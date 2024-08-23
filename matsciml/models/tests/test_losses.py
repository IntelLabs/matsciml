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
