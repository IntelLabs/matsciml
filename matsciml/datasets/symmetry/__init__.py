from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path
from warnings import warn

_has_rowan = find_spec("rowan") is not None

if not _has_rowan:
    warn(
        f"`rowan` dependency was not installed. To generate the symmetry dataset, please install matsciml with `pip install './[symmetry]'`.",
    )

symmetry_devset = Path(__file__).parents[0].joinpath("devset")

from matsciml.datasets.symmetry.dataset import SyntheticPointGroupDataset
