from __future__ import annotations

from pathlib import Path

from matsciml.datasets.alexandria.dataset import AlexandriaDataset, M3GAlexandriaDataset

alexandria_devset = Path(__file__).parents[0].joinpath("devset")
