from __future__ import annotations

from logging import getLogger

# determine if intel libraries are available
from matsciml.common.packages import package_registry  # noqa: F401

__version__ = "1.1.0"

logger = getLogger(__file__)
