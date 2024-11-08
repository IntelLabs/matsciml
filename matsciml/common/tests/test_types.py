from __future__ import annotations

import torch
import pytest
from pydantic import ValidationError

from matsciml.common import types


def test_output_validation():
    """Simple unit test to make sure a minimal configuration works."""
    types.ModelOutput(batch_size=16, embeddings=None, node_energies=None)


def test_invalid_embeddings():
    """Type invalid embeddings"""
    with pytest.raises(ValidationError):
        types.ModelOutput(batch_size=8, embeddings="aggaga")


def test_valid_embeddings():
    embeddings = types.Embeddings(
        system_embedding=torch.rand(64, 32), point_embedding=torch.rand(162, 32)
    )
    types.ModelOutput(batch_size=64, embeddings=embeddings)


def test_incorrect_force_shape():
    """This passes a force tensor with too many dimensions"""
    with pytest.raises(ValidationError):
        types.ModelOutput(batch_size=8, forces=torch.rand(32, 4, 3))


def test_consistency_check_pass():
    types.ModelOutput(
        batch_size=8, forces=torch.rand(32, 3), node_energies=torch.rand(32, 1)
    )


def test_consistency_check_fail():
    with pytest.raises(RuntimeError):
        # check mismatching node energies and forces
        types.ModelOutput(
            batch_size=8, forces=torch.rand(32, 3), node_energies=torch.rand(64, 1)
        )
    with pytest.raises(RuntimeError):
        # check mismatch in
        types.ModelOutput(batch_size=4, total_energy=torch.rand(16, 1))
