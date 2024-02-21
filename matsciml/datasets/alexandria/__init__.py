from __future__ import annotations

from pathlib import Path
from matsciml.datasets.alexandria.api import AlexandriaRequest
from matsciml.datasets.alexandria.dataset import AlexandriaDataset

alexandria_devset = Path(__file__).parents[0].joinpath("devset")

__all__ = [
    "AlexandriaDataset",
    "AlexandriaRequest",
]
