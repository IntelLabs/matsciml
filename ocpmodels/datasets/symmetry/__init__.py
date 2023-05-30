
from warnings import warn
from pathlib import Path

_has_rowan = False

try:
    import rowan
    _has_rowan = True
except ImportError:
    warn(f"`rowan` dependency was not installed. To use symmetry dataset, please install matsciml with `pip install './[symmetry]'`.")

symmetry_devset = Path(__file__).parents[0].joinpath("devset")
