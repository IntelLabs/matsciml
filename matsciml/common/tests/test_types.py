from __future__ import annotations

from matsciml.common import types


def test_output_validation():
    """Simple unit test to make sure a minimal configuration works."""
    types.ModelOutput(batch_size=16, embeddings=None, node_energies=None)
