"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from loguru import logger

from matsciml.common.packages import package_registry

if "pyg" in package_registry:
    _has_pyg = True
else:
    _has_pyg = False

# load models if we have PyG installed
if _has_pyg:
    from matsciml.models.pyg.egnn import EGNN
    from matsciml.models.pyg.mace import MACE, ScaleShiftMACE, MACEWrapper
    from matsciml.models.pyg.faenet import FAENet

    __all__ = ["CGCNN", "EGNN", "FAENet", "MACE", "ScaleShiftMACE", "MACEWrapper"]

    # these packages need additional pyg dependencies
    if package_registry["torch_sparse"] and package_registry["torch_scatter"]:
        from matsciml.models.pyg.dimenet import DimeNetWrap  # noqa: F401
        from matsciml.models.pyg.dimenet_plus_plus import DimeNetPlusPlusWrap  # noqa: F401

        __all__.extend(["DimeNetWrap", "DimeNetPlusPlusWrap"])
    else:
        logger.warning(
            "Missing torch_sparse and torch_scatter; DimeNet models will not be available."
        )
    if package_registry["torch_scatter"]:
        from matsciml.models.pyg.forcenet import ForceNet  # noqa: F401
        from matsciml.models.pyg.schnet import SchNetWrap  # noqa: F401
        from matsciml.models.pyg.cgcnn import CGCNN  # noqa: F401

        __all__.extend(["ForceNet", "SchNetWrap", "CGCNN"])
    else:
        logger.warning(
            "Missing torch_scatter; ForceNet, SchNet, and FAENet models will not be available."
        )
