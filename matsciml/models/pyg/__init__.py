"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
from __future__ import annotations

from matsciml.common.packages import package_registry

if "pyg" in package_registry:
    _has_pyg = True
else:
    _has_pyg = False

# load models if we have PyG installed
if _has_pyg:
    from matsciml.models.pyg.cgcnn import CGCNN
    from matsciml.models.pyg.dimenet import DimeNetWrap
    from matsciml.models.pyg.dimenet_plus_plus import DimeNetPlusPlusWrap
    from matsciml.models.pyg.egnn import EGNN
    from matsciml.models.pyg.faenet import FAENet
    from matsciml.models.pyg.forcenet import ForceNet
    from matsciml.models.pyg.mace import MACE, ScaleShiftMACE
    from matsciml.models.pyg.schnet import SchNetWrap
