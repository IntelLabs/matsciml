from __future__ import annotations

from logging import getLogger

# determine if intel libraries are available
from matsciml.common.packages import package_registry

__version__ = "1.1.0"

logger = getLogger(__file__)


if package_registry["ipex"]:
    try:
        import intel_extension_for_pytorch  # noqa: F401
    except ImportError as e:
        logger.warning(f"Unable to load IPEX because of {e} - XPU may not function.")
if package_registry["ccl"]:
    import oneccl_bindings_for_pytorch  # noqa: F401
