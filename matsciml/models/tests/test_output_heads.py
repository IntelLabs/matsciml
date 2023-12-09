from __future__ import annotations

from itertools import product

import pytest
import torch
from e3nn import o3

from matsciml.common.registry import registry
from matsciml.models.common import IrrepOutputBlock, OutputBlock, OutputHead


@pytest.mark.parametrize("classname", ("OutputBlock", "IrrepOutputBlock"))
def test_block_in_registry(classname):
    block_type = registry.get_model_class(classname)
    if not block_type:
        raise NameError(f"{classname} was not found in registry.")


output_param_space = list(
    product(
        [None, True],
        ["torch.nn.SiLU", "torch.nn.ReLU", "torch.nn.GELU"],
        [False, True],
        [2, 64, 256],
        [2, 64, 256],
    ),
)


@pytest.fixture(params=output_param_space)
def output_block_fixture(request):
    norm, activation, residual, output_dim, input_dim = request.param
    if norm:
        norm = torch.nn.BatchNorm1d(output_dim)
    block = OutputBlock(
        output_dim=output_dim,
        input_dim=input_dim,
        norm=norm,
        activation=activation,
        residual=residual,
    )
    rand_in = torch.rand(8, input_dim)
    rand_out = torch.rand(8, output_dim)
    return (block, rand_in, rand_out)


@pytest.mark.dependency()
def test_output_block(output_block_fixture):
    block, rand_in, rand_out = output_block_fixture
    residual_test = bool(block.residual * (rand_in.shape != rand_out.shape))
    with torch.inference_mode():
        if residual_test:
            with pytest.raises(AssertionError):
                pred = block(rand_in)
        else:
            pred = block(rand_in)
            assert pred.shape == rand_out.shape


irrep_param_space = list(
    product(
        [None, True],
        ["torch.nn.SiLU", "torch.nn.ReLU"],
        [False, True],
        [2, 8, 16],
        [2, 8, 16],
    ),
)


@pytest.fixture(params=irrep_param_space)
def irrep_block_fixture(request):
    norm, activation, residual, output_dim, input_dim = request.param
    input_irrep = o3.Irreps(f"{input_dim}x0e + {input_dim}x1e")
    output_irrep = o3.Irreps(f"{output_dim}x0e + {output_dim}x1e")
    rand_in = input_irrep.randn(4, -1)
    rand_out = output_irrep.randn(4, -1)
    block = IrrepOutputBlock(
        output_dim=output_irrep,
        input_dim=input_irrep,
        norm=norm,
        activation=[activation, None],
        residual=residual,
    )
    block._output_irrep = output_irrep
    block._input_irrep = input_irrep
    return (block, rand_in, rand_out)


@pytest.mark.dependency()
def test_irrep_block(irrep_block_fixture):
    block, rand_in, rand_out = irrep_block_fixture
    residual_test = bool(block.residual * (block._output_irrep != block._input_irrep))
    with torch.inference_mode():
        if isinstance(block.layers[-1], torch.nn.BatchNorm1d):
            assert rand_out.size(-1) == block.layers[-1].weight.size(-1)
        if residual_test:
            with pytest.raises(AssertionError):
                pred = block(rand_in)
        else:
            pred = block(rand_in)
            assert pred.shape == rand_out.shape


@pytest.mark.parametrize(
    "head_kwargs",
    [
        {
            "output_dim": 16,
            "hidden_dim": 64,
            "input_dim": 8,
            "activation": "torch.nn.SiLU",
            "norm": None,
            "residual": False,
            "block_type": "OutputBlock",
        },
        {
            "output_dim": o3.Irreps("10x0e + 10x1e"),
            "input_dim": o3.Irreps("5x0e + 10x1e"),
            "hidden_dim": o3.Irreps("20x0e + 20x1e"),
            "activation": [torch.nn.SiLU(), None],
            "norm": True,
            "residual": False,
            "block_type": "IrrepOutputBlock",
        },
    ],
)
@pytest.mark.dependency(depends=["test_output_block", "test_irrep_block"])
def test_regular_output_head(head_kwargs):
    head = OutputHead(**head_kwargs)
    if head_kwargs["block_type"] == "IrrepOutputBlock":
        rand_in = head_kwargs["input_dim"].randn(4, -1)
        rand_out = head_kwargs["output_dim"].randn(4, -1)
    else:
        rand_in = torch.randn(4, head_kwargs["input_dim"])
        rand_out = torch.randn(4, head_kwargs["output_dim"])
    with torch.inference_mode():
        pred = head(rand_in)
        assert pred.shape == rand_out.shape
