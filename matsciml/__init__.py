from __future__ import annotations

__version__ = "1.1.0"

# determine if intel libraries are available
from matsciml.common.packages import package_registry

if package_registry["ipex"]:
    import intel_extension_for_pytorch  # noqa: F401
if package_registry["ccl"]:
    import oneccl_bindings_for_pytorch  # noqa: F401
