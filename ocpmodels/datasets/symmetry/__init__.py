
from warnings import warn
from pathlib import Path
from importlib.util import find_spec

_has_rowan = find_spec("rowan") is not None

if not _has_rowan:
    warn(f"`rowan` dependency was not installed. To generate the symmetry dataset, please install matsciml with `pip install './[symmetry]'`.")

symmetry_devset = Path(__file__).parents[0].joinpath("devset")
