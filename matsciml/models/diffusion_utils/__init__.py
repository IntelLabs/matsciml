from __future__ import annotations

__all__ = [
    "ATOMIC_RADII",
    "KHOT_EMBEDDINGS",
    "CONTINUOUS_EMBEDDINGS",
    "MAX_ATOMIC_NUM",
]

from matsciml.models.diffusion_utils.atomic_radii import ATOMIC_RADII
from matsciml.models.diffusion_utils.continuous_embeddings import CONTINUOUS_EMBEDDINGS
from matsciml.models.diffusion_utils.khot_embeddings import KHOT_EMBEDDINGS

MAX_ATOMIC_NUM = 100
