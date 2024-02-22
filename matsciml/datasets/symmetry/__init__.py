from __future__ import annotations

from importlib.util import find_spec
from warnings import warn

from matsciml.datasets.symmetry.dataset import SyntheticPointGroupDataset

_has_rowan = find_spec("rowan") is not None

if not _has_rowan:
    warn(
        "`rowan` dependency was not installed. To generate the symmetry dataset, please install matsciml with `pip install './[symmetry]'`.",
    )
