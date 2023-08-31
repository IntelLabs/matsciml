"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

try:
    import torch_geometric

    _has_pyg = True
except ImportError:
    _has_pyg = False

# load models if we have PyG installed
if _has_pyg:
    from ocpmodels.models.pyg.dimenet_plus_plus import DimeNetPlusPlusWrap
    from ocpmodels.models.pyg.dimenet import DimeNetWrap
    from ocpmodels.models.pyg.schnet import SchNetWrap
    from ocpmodels.models.pyg.forcenet import ForceNet
    from ocpmodels.models.pyg.cgcnn import CGCNN
