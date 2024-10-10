from __future__ import annotations

from loguru import logger
from importlib import util, import_module

from packaging.version import parse
from pkg_resources import DistributionNotFound, get_distribution

logger = logger.bind(name="matsciml.packages")

"""
The `package_registry` object is a convenient way to determine which packages have
been installed.
"""
package_registry = {}
package_registry["ipex"] = (
    True if util.find_spec("intel_extension_for_pytorch") else False
)
package_registry["ccl"] = (
    True if util.find_spec("oneccl_bindings_for_pytorch") else False
)
# graph specific packages; slightly more involved because we should try import
for package in ["torch_geometric", "torch_scatter", "torch_sparse", "dgl"]:
    success = False
    try:
        import_module(package)
        success = True
    except Exception:
        logger.opt(exception=True).warning(
            f"Could not import {package}, which may impact functionality."
        )
    package_registry[package] = success
# for backwards compatibility and looks better anyway
package_registry["pyg"] = package_registry["torch_geometric"]
package_registry["codecarbon"] = True if util.find_spec("codecarbon") else False


def get_package_version(module_name: str) -> str:
    """
    Programmatically retrieve the version of a Python package.

    Parameters
    ----------
    module_name : str
        Name of the package as if it were to be imported

    Returns
    -------
    str
        Version string of the package

    Raises
    ------
    ModuleNotFoundError:
        If the package is not installed
    """
    try:
        version = get_distribution(module_name)
        return version.version
    except DistributionNotFound:
        raise ModuleNotFoundError(f"Package {module_name} not found.")


def is_package_version_greater(module_name: str, target_version: str) -> bool:
    """
    Returns whether or not the version of a module is greater than the specified value.

    Parameters
    ----------
    module_name : str
        Module to check the version of.
    target_version : str
        Version to check if the installed package is newer.

    Returns
    -------
    bool
        True if the installed version if greater than the target version
    """
    actual_version = parse(get_package_version(module_name))
    target_version = parse(target_version)
    return actual_version > target_version
