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


def test_incorrect_force_shape():
    """This passes a force tensor with too many dimensions"""
    with pytest.raises(ValidationError):
        types.ModelOutput(batch_size=8, forces=torch.rand(32, 4, 3))
