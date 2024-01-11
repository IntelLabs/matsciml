from __future__ import annotations

import pytest
import torch

from matsciml.models import GalaPotential


@pytest.fixture
def gala_kwargs():
    # return a stock configuration for Gala
    model_args = {
        "D_in": 100,
        "hidden_dim": 32,
        "depth": 2,
        "merge_fun": "concat",
        "join_fun": "concat",
        "invariant_mode": "full",
        "covariant_mode": "full",
        "include_normalized_products": True,
        "invar_value_normalization": "momentum",
        "eqvar_value_normalization": "momentum_layer",
        "value_normalization": "layer",
        "score_normalization": "layer",
        "block_normalization": "layer",
        "equivariant_attention": False,
        "tied_attention": True,
        "encoder_only": True,
    }
    return model_args


@pytest.fixture
def data():
    data = {
        "pos": torch.vstack(
            [
                torch.rand(3, 3, requires_grad=True),
                torch.rand(3, 3, requires_grad=True),
                torch.rand(4, 3, requires_grad=True),
            ],
        ),
        # this should be 3 point clouds
        "pc_features": torch.rand(3, 4, 4, 200),
        "src_nodes": [torch.arange(3), torch.arange(3), torch.arange(4)],
        "dst_nodes": [torch.arange(3), torch.arange(3), torch.arange(4)],
        "sizes": [3, 3, 4],
    }
    return data


@pytest.fixture
def encoder_only():
    return {"encoder_only": True}


@pytest.fixture()
def not_encoder_only():
    return {"encoder_only": False}


@pytest.fixture()
def encoder_only_hidden_dim():
    return {"encoder_only": True, "hidden_dim": 128}


@pytest.fixture()
def not_encoder_only_hidden_dim():
    return {"encoder_only": False, "hidden_dim": 128}


@pytest.mark.parametrize(
    "config, shape",
    [
        ("not_encoder_only", (3, 1)),
        ("encoder_only", (3, 32)),
        ("encoder_only_hidden_dim", (3, 128)),
        ("not_encoder_only_hidden_dim", (3, 1)),
    ],
)
def test_gala_config_no_grad(config, shape, request):
    stock_config = request.getfixturevalue("gala_kwargs")
    new_config = request.getfixturevalue(config)
    # overwrite config with new parameters
    stock_config.update(**new_config)
    model = GalaPotential(**stock_config)
    data = request.getfixturevalue("data")
    with torch.no_grad():
        output = model(data)
    assert output.shape == shape


@pytest.mark.parametrize(
    "config, shape",
    [
        ("not_encoder_only", (3, 1)),
        ("encoder_only", (3, 32)),
        ("encoder_only_hidden_dim", (3, 128)),
        ("not_encoder_only_hidden_dim", (3, 1)),
    ],
)
def test_gala_config_grad(config, shape, request):
    stock_config = request.getfixturevalue("gala_kwargs")
    new_config = request.getfixturevalue(config)
    # overwrite config with new parameters
    stock_config.update(**new_config)
    model = GalaPotential(**stock_config)
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    data = request.getfixturevalue("data")
    # run a fake training step
    opt.zero_grad()
    output = model(data)
    assert output.shape == shape
    fake_target = torch.rand_like(output)
    loss = torch.nn.functional.mse_loss(output, fake_target)
    assert torch.isfinite(loss)
    loss.backward()
    opt.step()


def test_gala_force_no_backprop(gala_kwargs, data):
    # predict an energy value so we can backprop
    gala_kwargs["encoder_only"] = False
    model = GalaPotential(**gala_kwargs)
    output = model(data)
    assert output.shape == (3, 1)
    assert hasattr(output, "grad_fn")
    force = (
        -1
        * torch.autograd.grad(
            output,
            data["pos"],
            grad_outputs=torch.ones_like(output),
            create_graph=True,
        )[0]
    )
    assert force.shape == (10, 3)
    model.zero_grad(set_to_none=True)


def test_gala_force_backprop(gala_kwargs, data):
    # predict an energy value so we can backprop
    gala_kwargs["encoder_only"] = False
    model = GalaPotential(**gala_kwargs)
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    opt.zero_grad()
    output = model(data)
    assert output.shape == (3, 1)
    assert hasattr(output, "grad_fn")
    force = (
        -1
        * torch.autograd.grad(
            output,
            data["pos"],
            grad_outputs=torch.ones_like(output),
            create_graph=True,
        )[0]
    )
    assert force.shape == (10, 3)
    target_force = torch.rand_like(force)
    loss = torch.nn.functional.mse_loss(force, target_force)
    loss.backward(retain_graph=True)
    loss.backward()
    opt.step()
    model.zero_grad(set_to_none=True)
